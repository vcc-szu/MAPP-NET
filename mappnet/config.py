import os
from datetime import datetime

class Config():
    def __init__(self):
        self.gpu_index=1
        os.environ["CUDA_VISIBLE_DEVICES"]=str(self.gpu_index)

        self.config_model()
        self.config_dataset()
        self.config_path()
    def config_model(self):
        self.DEBUG=False
        self.model_name='mappnet'
        self.motion_type='r'
        self.motion_param_dim = 6
        self.pc_channel = 3
        self.num_point = 1024
        self.num_frame = 5
        self.num_frame_classes = 10
        self.batch_size = 32
        self.use_ground_truth_seg = True
        self.use_pred_mo = True
    def config_dataset(self):
        self.cache_size = 1500000
        self.use_scan_dataset = False
        if self.use_scan_dataset:
            self.dataset_suffix = ''
            self.sample_per_unit = 1
        else:
            self.dataset_suffix = ''
        self.sample_per_unit = 3
    def config_path(self):
        # general setting
        self.this_file_path = os.path.dirname(os.path.realpath(__file__))
        #self.this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.project_root_path = os.path.abspath(os.path.join(self.this_file_path, '../')) + '/'

        #self.project_root_path = '/home/xieke/zihao/Monet/'
        #self.project_root_path = os.path.abspath(os.path.join(self.this_file_path, '../../'))+'/'
        
        self.output_path = self.project_root_path + 'output/'+ self.model_name + '/'
        self.data_path = self.project_root_path + 'data/'
        self.code_path = self.project_root_path + 'src/'
        #self.pairnetcode_path = self.code_path + model_name + '/'
        self.modelcode_path = self.code_path + '/models/'


        self.data_config_path=self.data_path+'datalist/'
        self.datalist_path = self.data_config_path
        self.pc_path=self.data_path+'point_fix_frame_{}{}/'.format(self.num_point, self.dataset_suffix)
        self.seg_path=self.data_path+'seg_fix_frame_{}{}/'.format(self.num_point, self.dataset_suffix)
        self.mo_path=self.data_path+'motion_{}{}/'.format(self.num_point, self.dataset_suffix)
        self.motion_path=self.mo_path

        self.train_datalist_path=os.path.join(self.data_config_path, 'train_{}_{}_{}{}.txt'.format(self.model_name, self.motion_type, self.num_point, self.dataset_suffix))
        self.test_datalist_path =os.path.join(self.data_config_path,  'test_{}_{}_{}{}.txt'.format(self.model_name, self.motion_type, self.num_point, self.dataset_suffix))

    # train setting
    def config_train(self, run_message='test'):
        self.resume_training = False
        self.runmsg=run_message
        self.max_epoch = 101
        self.train_batch_size = 8
        self.train_gpu_index= self.gpu_index
        self.save_name = '%s_%s_%s' % (self.model_name, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), self.motion_type)
        if self.DEBUG==True:
            self.save_name = 'DEBUG'
        assert(self.runmsg!='')
        self.save_name+='_{}'.format(self.runmsg)
        self.save_path = self.output_path + self.save_name + '/'
        self.ckpt_path = self.save_path + 'ckpts/'
        self.summary_path = self.save_path + 'summary/'
        self.train_summary_path = self.summary_path + 'train'
        if not os.path.exists( self.save_path ): os.makedirs( self.save_path )
        if not os.path.exists(self.train_summary_path): os.makedirs(self.train_summary_path)
        if not os.path.exists(self.ckpt_path): os.makedirs(self.ckpt_path)
        
        self.resume_model_name='pairnet_2019-01-05-00-19-17_R_staged_training'
        self.resume_load_path=self.output_path + self.resume_model_name + '/'
        self.resume_ckpt_index = 40
        self.resume_model_path=self.resume_load_path+'ckpts/model.ckpt-{}'.format(self.resume_ckpt_index)

    # test setting
    def config_test(self, saved_name='pairnet_2019-01-05-11-42-10_R_staged_tr_continued_staged_100', ckpt_index=100):
        if saved_name is None:
            saved_name = self.save_name
        self.test_batch_size = 1
        self.test_split = 'test'
        self.load_path=self.output_path + saved_name + '/'
        self.model_path=self.load_path+'ckpts/model.ckpt-{}'.format(ckpt_index)

        self.summary_path = self.load_path + 'summary/'
        self.test_summary_path = self.summary_path + 'test' + datetime.now().strftime('%Y%m%d%H%M%S')

        self.eval_path = self.load_path+'eval_{}/'.format(self.test_split)
        if not os.path.exists(self.eval_path): os.mkdir(self.eval_path)
        self.predict_save_path = self.eval_path + 'predict/'
        if not os.path.exists(self.predict_save_path): os.mkdir(self.predict_save_path)
        self.stat_path = self.eval_path+'statistics/'
        if not os.path.exists(self.stat_path): os.mkdir(self.stat_path)
        self.test_gpu_index=self.gpu_index

    # visualization setting
    def config_visual(self):
        self.config_test()
        self.visual_split = self.test_split
        self.vis_path = self.eval_path + 'visual/'
        self.visinfo_path = self.eval_path + 'visual_info/'
        self.visgif_path = self.eval_path + 'visual_gif/'
        self.visvideo_path = self.eval_path + 'visual_video/'
        if not os.path.exists( self.vis_path ): os.makedirs( self.vis_path )
        if not os.path.exists( self.visinfo_path ): os.makedirs( self.visinfo_path )
        if not os.path.exists( self.visgif_path ): os.makedirs( self.visgif_path )
        #if not os.path.exists( self.visinfo_path ): os.makedirs( self.visinfo_path )
    
    def config_paper(self):
        self.paper_path = self.project_root_path + 'output/paper/'
        self.cam_n = 10
        self.paper_result_model_names =  ['rnnmodel_segregcls_pc_test_2019-01-11-16-38-57_flex_10cam_rec_disp_1seg_1reg_cls_R_T',
            'rnnmodel_segregcls_pc_test_2019-01-11-16-44-34_flex_4cam_rec_disp_1seg_1reg_cls_R_T',
            'rnnmodel_segregcls_pc_2019-01-08-00-36-31_flex_4cam_rec_disp_1seg_1reg_cls_R_T',
            'rnnmodel_segregcls_pc_2019-01-08-00-30-47_flex_10cam_rec_disp_1seg_1reg_cls_R_T',
            'rnnmodel_segregcls_pc_test_2019-01-11-14-20-28_flex_rec_disp_1seg_1reg_cls_R_T_120',
            'rnnmodel_segreg_pc_2019-01-10-21-52-25_flex_toend_rec_disp_seg_reg_R',
            'rnnmodel_segreg_pc_2019-01-04-23-45-46_flex_chamfer_seg_reg_T',
            'rnnmodel_segreg_pc_2019-01-04-16-11-43_flex_chamfer_seg_reg_R']
        self.paper_result_model_names = ['rnnmodel_segregcls_pc_test_2019-01-11-16-38-57_flex_10cam_rec_disp_1seg_1reg_cls_R_T']
        if self.cam_n==4:
                self.basenet_info_path = self.project_root_path + 'output/baseNet/basemodel_2019-01-12-00-45-55_R_T_4cam/eval/info.txt'
                self.paper_result_model_names = ['rnnmodel_segregcls_pc_test_2019-01-11-16-44-34_flex_4cam_rec_disp_1seg_1reg_cls_R_T']

        elif self.cam_n==10:
                self.basenet_info_path = self.project_root_path + 'output/baseNet/basemodel_2019-01-12-00-44-18_R_T_10cam/eval/info.txt'
                self.paper_result_model_names = ['rnnmodel_segregcls_pc_test_2019-01-11-16-38-57_flex_10cam_rec_disp_1seg_1reg_cls_R_T']

        
        self.output_num_per_model = 2

config=Config()
