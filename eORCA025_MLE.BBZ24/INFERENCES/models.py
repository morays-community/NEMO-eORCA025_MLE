import numpy as np
import sys
import torch


# ============================= #
# -- User Defined Parameters --
# ============================= #
# res_string can be one of the following ['1_12','1_8','1_4','1_2','1']
res_string = '1_4'
model_path = '/gpfswork/rech/cli/udp79td/local_libs/morays/NEMO-MLE_Fluxes/MLE-Fluxes.CNN/INFERENCES/NEMO_MLE/trained_models'
norm_path = '/gpfswork/rech/cli/udp79td/local_libs/morays/NEMO-MLE_Fluxes/MLE-Fluxes.CNN/INFERENCES/norms'

# ================================= #
# --------- DO NOT MODIFY --------
# ================================= #
norms = { 'means' : {}, 'devs' : {} }
input_string = ['grad_B','FCOR' , 'HML', 'TAU', 'Q', 'div', 'vort', 'strain']

# (re)normalization values
norms['means']['WB_sg'] = np.load( norm_path + '/norm_' + res_string + '/WB_sg_mean.npy' )
norms['devs']['WB_sg'] = np.load( norm_path + '/norm_' + res_string + '/WB_sg_std.npy' )

for name in input_string:
    file_mean = norm_path + '/norm_' + res_string + '/' + name + '_mean.npy'
    file_dev = norm_path + '/norm_' + res_string +  '/' + name  + '_std.npy'
    norms['means'][name] = np.load(file_mean) 
    norms['devs'][name] = np.load(file_dev)

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

# load model once: in-repo test or in local deployed config dir
try:
    net = torch.load( 'fcnn_k5_l7_m_HBL_res_'+res_string+'.pt' , map_location=device)
except:
    net = torch.load( model_path + '/fcnn_k5_l7_m_HBL_res_'+res_string+'.pt' , map_location=device)


#       Utils 
# -----------------
def Is_None(*inputs):
    """ Test presence of at least one None in inputs """
    return any(item is None for item in inputs)


#       Main Model Routines
# ------------------------------
@torch.no_grad()
def vert_buoyancy_flux_CNN(*inputs, tmask):
    """ Compute vertical buoyancy flux with pre-trained CNN following Bodner et al (2024) """
    if Is_None(*inputs[0]):
        return None
    else:
        # load global values
        global res_string, model_path, norms, net, device
        net.eval()

        # normalize and mask inputs
        to_stack = []
        for name, arr in zip(input_string,inputs[0]):
            mean = norms['means'][name]
            dev = norms['devs'][name]
            work_arr = ( arr - mean ) / dev * tmask
            to_stack.append( work_arr[:,:,0] )

        # build batch
        x_data = np.stack( to_stack, axis=0 )
        x_data = x_data[ np.newaxis, ... ]
        x_data = torch.from_numpy( x_data ).to( device, dtype=torch.float32 )

        # passing the entire batch in test_loader into the CNN to get prediction of w'b'              
        if device.type == 'cuda':
            w_b = net( x_data.to(device) ).detach().cpu().numpy()
        else:
            w_b = net( x_data.to(device) ).detach().numpy() 

        # renormalize
        mean = norms['means']['WB_sg']
        dev = norms['devs']['WB_sg']
        w_b = np.squeeze( w_b )
        w_b = w_b[ : , : , np.newaxis ]
        w_b = ( w_b * tmask * dev ) + mean

        return -0.001*w_b #-0.1*w_b


if __name__ == '__main__' :
    gradb = np.random.rand(120,100,1).astype('float32')
    fcor = np.random.rand(120,100,1).astype('float32')
    hml = np.random.rand(120,100,1).astype('float32')
    tau = np.random.rand(120,100,1).astype('float32')
    q = np.random.rand(120,100,1).astype('float32')
    div = np.random.rand(120,100,1).astype('float32')
    vort = np.random.rand(120,100,1).astype('float32')
    strain = np.random.rand(120,100,1).astype('float32')
    msk = np.ones((120,100,1)).astype('float32')
    vert_flux = vert_buoyancy_flux_CNN((gradb,fcor,hml,tau,q,div,vort,strain),tmask=msk)
    print(f'Returned vert_flux : {vert_flux.shape}')
    print(f'Test successful')
