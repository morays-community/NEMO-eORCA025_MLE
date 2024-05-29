import numpy as np
import sys
import torch

#import submeso_ml.systems.regression_system as regression_system
#import submeso_ml.models.fcnn as fcnn
#import submeso_ml.data.dataset as dataset


# ============================= #
# -- User Defined Parameters --
# ============================= #
# res_string can be one of the following ['1_12','1_8','1_4','1_2','1']
res_string = '1_4'
rcpt_fld_size = 7
model_path = '/gpfswork/rech/cli/udp79td/local_libs/morays/NEMO-MLE_Fluxes/MLE-Fluxes.CNN/INFERENCES/NEMO_MLE/trained_models'

# ================================= #
# --------- DO NOT MODIFY --------
# ================================= #
norms = { 'means' : {}, 'devs' : {} }
input_string = ['grad_B','FCOR' , 'HML', 'TAU', 'Q', 'div', 'vort', 'strain']

# (re)normalization values
norms['means']['WB_sg'] = np.load( model_path + '/norm_' + res_string + '/WB_sg_mean.npy' )
norms['devs']['WB_sg'] = np.load( model_path + '/norm_' + res_string + '/WB_sg_std.npy' )

for name in input_string:
    file_mean = model_path + '/norm_' + res_string + '/' + name + '_mean.npy'
    file_dev = model_path + '/norm_' + res_string +  '/' + name  + '_std.npy'
    norms['means'][name] = np.load(file_mean) 
    norms['devs'][name] = np.load(file_dev)

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')


#       Utils 
# -----------------
def Is_None(*inputs):
    """ Test presence of at least one None in inputs """
    return any(item is None for item in inputs)


def make_edges( array , edge ):

    # build north edge
    folded_halos = array[ :, 1:1+edge , : ].copy()
    folded_halos = np.flip( folded_halos )
    folded_halos = np.roll( folded_halos , 1 , axis=0 )
    edged_array = np.hstack( (folded_halos,array) )

    # build east-west edges
    left = edged_array[:edge,:,:].copy()
    right = edged_array[-edge:,:,:].copy()
    edged_array = np.vstack( (right,edged_array)  )
    edged_array = np.vstack( (edged_array,left) )

    # build south edge
    up = edged_array[:,:edge,:].copy()
    up = up * 0.0
    edged_array = np.hstack( (edged_array,up)  )

    return edged_array


#       Main Model Routines
# ------------------------------
@torch.no_grad()
def vert_buoyancy_flux_CNN(*inputs, tmask):
    """ Compute vertical buoyancy flux with pre-trained CNN following Bodner et al (2024) """
    if Is_None(*inputs[0]):
        return None
    else:
        # load global values
        global res_string, model_path, norms, rcpt_fld_size
        edge_size = rcpt_fld_size // 2

        # load pre-trained model without HBL
        model = torch.load( model_path+'/fcnn_k5_l7_m_HBL_res_'+res_string+'.pt' )
        model.eval()

        # normalize and mask inputs
        to_stack = []
        for name, arr in zip(input_string,inputs[0]):
            mean = norms['means'][name]
            dev = norms['devs'][name]
            edged_arr = ( arr - mean ) / dev * tmask
            edged_arr = make_edges( edged_arr , edge_size )
            to_stack.append( edged_arr[:,:,0] )

        # build batch
        x_data = np.stack( to_stack, axis=0 )
        x_data = x_data[ np.newaxis, ... ]
        x_data = torch.from_numpy( x_data ).to( dtype=torch.float32 )

        # passing the entire batch in test_loader into the CNN to get prediction of w'b'                
        w_b = model( x_data.to(device) ).detach().numpy() 
        # renormalize
        mean = norms['means']['WB_sg']
        dev = norms['devs']['WB_sg']
        w_b = np.squeeze( w_b )
        w_b = w_b[ edge_size:-edge_size , edge_size:-edge_size , np.newaxis ] # ?? ?? 
        w_b = ( w_b * tmask * dev ) + mean

        return w_b
