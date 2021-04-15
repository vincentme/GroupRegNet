import model.regnet, model.loss, model.util, utils.structure
import torch, os
import SimpleITK as sitk
import matplotlib.pyplot as plt; plot_dpi = 300
import numpy as np
import logging, tqdm
logging.basicConfig(level=logging.INFO, format = '%(levelname)s: %(message)s')
from scipy import interpolate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

case = 1
crop_range = [slice(0, 83), slice(43, 200), slice(10, 250)]
pixel_spacing = np.array([0.97, 0.97, 2.5], dtype = np.float32)


# case = 2
# crop_range = [slice(5, 98), slice(30, 195), slice(8, 243)]
# pixel_spacing = np.array([1.16, 1.16, 2.5], dtype = np.float32)


# case = 3
# crop_range = [slice(0, 95), slice(42, 209), slice(10, 248)]
# pixel_spacing = np.array([1.15, 1.15, 2.5], dtype = np.float32)


# case = 4
# crop_range = [slice(0, 90), slice(45, 209), slice(11, 242)]
# pixel_spacing = np.array([1.13, 1.13, 2.5], dtype = np.float32)


# case = 5
# crop_range = [slice(0, 90), slice(60, 222), slice(16, 237)]
# pixel_spacing = np.array([1.10, 1.10, 2.5], dtype = np.float32)


# case = 6
# crop_range = [slice(10, 107), slice(144, 328), slice(132, 426)]
# pixel_spacing = np.array([0.97, 0.97, 2.5], dtype = np.float32)


# case = 7
# crop_range = [slice(13, 108), slice(141, 331), slice(114, 423)]
# pixel_spacing = np.array([0.97, 0.97, 2.5], dtype = np.float32)


# case = 8
# crop_range = [slice(18, 118), slice(84, 299), slice(113, 390)]
# pixel_spacing = np.array([0.97, 0.97, 2.5], dtype = np.float32)


# case = 9
# crop_range = [slice(0, 70), slice(126, 334), slice(128, 390)]
# pixel_spacing = np.array([0.97, 0.97, 2.5], dtype = np.float32)


# case = 10
# crop_range = [slice(0, 90), slice(119, 333), slice(140, 382)]
# pixel_spacing = np.array([0.97, 0.97, 2.5], dtype = np.float32)



data_folder = f'/data/dirlab/Case{case}Pack/Image_MHD/'
landmark_file = f'/data/dirlab/Case1Pack/ExtremePhases/case{case}_00_50.pt'
states_folder = '/result/general_reg/dirlab/'
config = dict(
dim = 3, # dimension of the input image
intensity_scale_const = 1000., # (image - intensity_shift_const)/intensity_scale_const
intensity_shift_const = 1000., 
# scale = 0.7,
scale = 0.5,
initial_channels = 32, 
depth = 4, 
max_num_iteration = 3000,
normalization = True, # whether use normalization layer
learning_rate = 1e-2,
smooth_reg = 1e-3,
cyclic_reg = 1e-2,
ncc_window_size = 5, 
load = False,
load_optimizer = False,
group_index_list = None,
pair_disp_indexes = [0, 5],
pair_disp_calc_interval = 20,
stop_std = 0.0007,
stop_query_len = 100,
)
config = utils.structure.Struct(**config)

landmark_info = torch.load(landmark_file)
landmark_disp = landmark_info['disp_00_50'] # w, h, d
landmark_00 = landmark_info['landmark_00']
landmark_50 = landmark_info['landmark_50']
crop_min = np.min(np.concatenate((landmark_00, landmark_50), axis = 0), axis = 0) - 8
crop_max = np.max(np.concatenate((landmark_00, landmark_50), axis = 0), axis = 0) + 8
print(crop_min)
print(crop_max)

image_file_list = sorted([file_name for file_name in os.listdir(data_folder) if file_name.lower().endswith('mhd')])
image_list = [sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_folder, file_name))) for file_name in image_file_list]
input_image = torch.stack([torch.from_numpy(image)[None] for image in image_list], 0)
if config.group_index_list is not None:
    input_image = input_image[config.group_index_list]

input_image = (input_image - config.intensity_shift_const)/config.intensity_scale_const

input_image = input_image[:, :, crop_range[0], crop_range[1], crop_range[2]]
image_shape = np.array(input_image.shape[2:]) # (d, h, w)
num_image = input_image.shape[0] # number of image in the group
regnet = model.regnet.RegNet_single(dim = config.dim, n = num_image, scale = config.scale, depth = config.depth, initial_channels = config.initial_channels, normalization = config.normalization)

ncc_loss = model.loss.NCC(config.dim, config.ncc_window_size)
regnet = regnet.to(device)
input_image = input_image.to(device)
ncc_loss = ncc_loss.to(device)
optimizer = torch.optim.Adam(regnet.parameters(), lr = config.learning_rate)
calcdisp = model.util.CalcDisp(dim = config.dim, calc_device = 'cuda')

if config.load:
    state_file = os.path.join(states_folder, config.load)
    if os.path.exists(state_file):
        state_file = os.path.join(states_folder, config.load)
        states = torch.load(state_file, map_location = device)
        regnet.load_state_dict(states['model'])
        if config.load_optimizer:
            optimizer.load_state_dict(states['optimizer'])
            logging.info(f'load model and optimizer state {config.load}.pth')
        else:
            logging.info(f'load model state {config.load}.pth')

grid_tuple = [np.arange(grid_length, dtype = np.float32) for grid_length in image_shape]
landmark_00_converted = np.flip(landmark_00, axis = 1) - np.array([crop_range[0].start, crop_range[1].start, crop_range[2].start], dtype = np.float32)


diff_stats = []
stop_criterion = model.util.StopCriterion(stop_std = config.stop_std, query_len = config.stop_query_len)
pbar = tqdm.tqdm(range(config.max_num_iteration))
for i in pbar:
    optimizer.zero_grad()
    res = regnet(input_image)
    
    total_loss = 0.
    if 'disp_i2t' in res:
        simi_loss = (ncc_loss(res['warped_input_image'], res['template']) + ncc_loss(input_image, res['warped_template']))/2.
    else:
        simi_loss = ncc_loss(res['warped_input_image'], res['template'])
    total_loss += simi_loss
    
    
    if config.smooth_reg > 0:
        if 'disp_i2t' in res:
            smooth_loss = (model.loss.smooth_loss(res['scaled_disp_t2i']) + model.loss.smooth_loss(res['scaled_disp_i2t']))/2.
        else:
            # smooth_loss = model.loss.smooth_loss(res['scaled_disp_t2i'])
            smooth_loss = model.loss.smooth_loss(res['scaled_disp_t2i'], res['scaled_template'])
        total_loss += config.smooth_reg*smooth_loss
        smooth_loss_item = smooth_loss.item()
    else:
        smooth_loss_item = 0
        
    if config.cyclic_reg > 0:
        if 'disp_i2t' in res:
            # cyclic_loss = (torch.mean((torch.sum(res['scaled_disp_t2i'], 0))**2) + torch.mean((torch.sum(res['scaled_disp_i2t'], 0)))**0.5)/2.
            cyclic_loss = ((torch.mean((torch.sum(res['scaled_disp_t2i'], 0))**2))**0.5 + (torch.mean((torch.sum(res['scaled_disp_i2t'], 0))**2))**0.5)/2.
        else:
            cyclic_loss = (torch.mean((torch.sum(res['scaled_disp_t2i'], 0))**2))**0.5
        total_loss += config.cyclic_reg*cyclic_loss
        cyclic_loss_item = cyclic_loss.item()
    else:
        cyclic_loss_item = 0
    
    total_loss.backward()
    optimizer.step()
    
    stop_criterion.add(simi_loss.item())
    if stop_criterion.stop():
        break

    pbar.set_description(f'{i}, simi. loss {simi_loss.item():.4f}, smooth loss {smooth_loss_item:.3f}, cyclic loss {cyclic_loss_item:.3f}')

    if i % config.pair_disp_calc_interval == 0:
        if 'disp_i2t' in res:
            disp_i2t = res['disp_i2t'][config.pair_disp_indexes]
        else:
            disp_i2t = calcdisp.inverse_disp(res['disp_t2i'][config.pair_disp_indexes])
        composed_disp = calcdisp.compose_disp(disp_i2t, res['disp_t2i'][config.pair_disp_indexes], mode = 'all')
        composed_disp_np = composed_disp.cpu().numpy() # (2, 2, 3, d, h, w)

        inter = interpolate.RegularGridInterpolator(grid_tuple, np.moveaxis(composed_disp_np[0, 1], 0, -1))
        calc_landmark_disp = inter(landmark_00_converted)

        diff = (np.sum(((calc_landmark_disp - landmark_disp)*pixel_spacing)**2, 1))**0.5
        diff_stats.append([i, np.mean(diff), np.std(diff)])
        print(f'\ndiff: {np.mean(diff):.2f}+-{np.std(diff):.2f}({np.max(diff):.2f})')

    
if 'disp_i2t' in res:
    disp_i2t = res['disp_i2t'][config.pair_disp_indexes]
else:
    disp_i2t = calcdisp.inverse_disp(res['disp_t2i'][config.pair_disp_indexes])
composed_disp = calcdisp.compose_disp(disp_i2t, res['disp_t2i'][config.pair_disp_indexes], mode = 'all')
composed_disp_np = composed_disp.cpu().numpy() # (2, 2, 3, d, h, w)
inter = interpolate.RegularGridInterpolator(grid_tuple, np.moveaxis(composed_disp_np[0, 1], 0, -1))
calc_landmark_disp = inter(landmark_00_converted)

diff = (np.sum(((calc_landmark_disp - landmark_disp)*pixel_spacing)**2, 1))**0.5
diff_stats.append([i, np.mean(diff), np.std(diff)])
print(f'\ndiff: {np.mean(diff):.2f}+-{np.std(diff):.2f}({np.max(diff):.2f})')
diff_stats = np.array(diff_stats)


res['composed_disp_np'] = composed_disp_np
states = {'config': config, 'model': regnet.state_dict(), 'optimizer': optimizer.state_dict(), 'registration_result':res, 'loss_list':stop_criterion.loss_list, 'diff_stats':diff_stats}
index = len([file for file in os.listdir(states_folder) if file.endswith('pth')])
states_file = f'reg_dirlab_case{case}_{index:03d}.pth'
torch.save(states, os.path.join(states_folder, states_file))

logging.info(f'save model and optimizer state {states_file}')


plt.figure(dpi = plot_dpi)
plt.plot(stop_criterion.loss_list, label = 'simi')
plt.title('similarity loss vs iteration')
