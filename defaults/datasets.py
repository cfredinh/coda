from utils import *
from .bases import BaseSet
from scipy.io import mmread
from torchvision.transforms import ToTensor, ToPILImage


DATA_INFO = {
              "JUMP_TARGET" : {"dataset_location": "JUMP_TARGET"},
}

# # # Cell Painting # # # Cell Painting # # # Cell Painting # # # Cell Painting
    
class JUMP_TARGET(BaseSet):
    
    img_channels = 3
    is_multiclass = True
    task = 'classification'
    mean = (0.415, 0.221, 0.073)
    std  = (0.275, 0.150, 0.081)
    moa_subset = True
    cross_batch = False
    int_to_labels = {0: 'temp'}
    target_metric = 'accuracy'
    knn_nhood = 50    
    n_classes = 54

    split_setting = "" #defined in params 
    test_subset   = ""
    test_plate    = ""
    split_setting_source = ""
    
    labels_to_ints = {val: key for key, val in int_to_labels.items()}
    
    def __init__(self, dataset_params, mode='train'):
        self.attr_from_dict(dataset_params)
        self.dataset_location = DATA_INFO["JUMP_TARGET"]["dataset_location"]
        self.mode = mode
        self.data, self.dataframe = self.get_data_as_list()
        self.transform, self.resizing = self.get_transforms()
        
    def get_data_as_list(self):
        data_list = []
        
        datainfo  = pd.read_csv(self.root_dir+ "metadata_Target_2_moa.csv", engine='python', index_col=0)
        if self.moa_subset:
            datainfo = datainfo.dropna(subset=["Metadata_moa"])
        
        self.int_to_labels = dict(zip(datainfo.moa_id, datainfo.Metadata_moa))
        self.labels_to_int = dict(zip(datainfo.Metadata_moa, datainfo.moa_id))

        labellist = datainfo.moa_id.tolist()
        img_names = (datainfo["Metadata_Plate"] + "/" + datainfo["paths"]).tolist()
        img_names = [os.path.join(self.root_dir, imname ) for imname in img_names]
        plate     = datainfo.Metadata_Plate.tolist()
        split     = datainfo.split.tolist()
        cmpd_id   = datainfo.JCP2022_id.tolist()
        d_source  = datainfo.Metadata_Source.tolist()
        s_batch   = datainfo.Metadata_Batch.tolist()
        
        dataframe = pd.DataFrame(list(zip(img_names, labellist, plate, split, cmpd_id, d_source, s_batch)), 
                                 columns=['img_path', 'label', 'plate', 'split', "cmpd_id", "source", "exp_batch"])
        
        if self.split_setting == "single_MICROSCOPE_specific":
            print("GETTING TEST DATA BELONGING TO SOURCE: ", self.test_subset)
            if self.split_setting_source == "source_3":     
                source_selected = "source_3" 
                if self.mode == 'train':
                    data = dataframe[(dataframe.source == source_selected) & (dataframe.plate != "JCPQC016")]
                elif self.mode in ['val', 'eval']:
                    data = dataframe[(dataframe.source == source_selected) & (dataframe.plate == "JCPQC016")]
                else:
                    data = dataframe[(dataframe.source.isin([self.test_subset]))]
            elif self.split_setting_source == "source_5":     
                source_selected = "source_5" 
                if self.mode == 'train':
                    data = dataframe[(dataframe.source == source_selected) & (dataframe.plate != "ACPJUM012")]
                elif self.mode in ['val', 'eval']:
                    data = dataframe[(dataframe.source == source_selected) & (dataframe.plate == "ACPJUM012")]
                else:
                    data = dataframe[(dataframe.source.isin([self.test_subset]))]
            elif self.split_setting_source == "source_8":     
                source_selected = "source_8" 
                if self.mode == 'train':
                    data = dataframe[(dataframe.source == source_selected) & (dataframe.plate != "A1166127")]
                elif self.mode in ['val', 'eval']:
                    data = dataframe[(dataframe.source == source_selected) & (dataframe.plate == "A1166127")]
                else:
                    data = dataframe[(dataframe.source.isin([self.test_subset]))]
            elif self.split_setting_source == "source_11":     
                source_selected = "source_11" 
                if self.mode == 'train':
                    data = dataframe[(dataframe.source == source_selected) & (dataframe.plate != "EC103-132LM2")]
                elif self.mode in ['val', 'eval']:
                    data = dataframe[(dataframe.source == source_selected) & (dataframe.plate == "EC103-132LM2")]
                else:
                    data = dataframe[(dataframe.source.isin([self.test_subset]))]
            else:
                print("NONE OF THE PRE SELECTED SUBSETS discovered")

            print("AVAILABLE DATA FROM SETTING: ", self.mode, "found as: ", data.shape)


        labels    = data['label'   ].values.tolist()
        img_paths = data['img_path'].values.tolist()
        plates    = data['plate'   ].values.tolist()
        cmpd_ids  = data['cmpd_id' ].values.tolist()
        
        data_list = [{'img_path': img_path, 'label': label, 'plate': plate, 'cmpd_id': cmpd_id, 'dataset': self.name}
                     for img_path, label, plate, cmpd_id in zip(img_paths, labels, plates, cmpd_ids)]

        dataframe = pd.DataFrame(data_list).drop_duplicates() 
        print("Size of dataset: ", dataframe.shape)
        return data_list, dataframe       
    
    def get_image_data(self, path: str) -> str:

        img = Image.open(path)

        img = np.array(img)
        img = np.transpose(np.array(np.hsplit(img,5))[[0,1,4]], axes=[1,2,0]) 
        
        return Image.fromarray(img)#.convert('RGB')
    
    def __getitem__(self, idx): 
        
        img_path = self.data[idx]['img_path']
        label    = torch.as_tensor(self.data[idx]['label'])
        
        img = self.get_image_data(img_path)
        
        if self.cross_batch and (self.mode == 'train'):
            
            cmp_id   = self.data[idx]['cmpd_id']
            plate_id = self.data[idx]['plate']

            cb_idx = self.dataframe[(self.dataframe.cmpd_id == cmp_id) & (self.dataframe.plate != plate_id) ].sample().index[0]
            
            cb_img_path = self.data[cb_idx]['img_path']
            cb_label    = torch.as_tensor(self.data[cb_idx]['label'])

            cb_img = self.get_image_data(cb_img_path)
            
            if self.resizing is not None:
                img    = self.resizing(img)
                cb_img = self.resizing(cb_img)

            if self.transform is not None:
                if isinstance(self.transform, list):
                    # devide the transform to do half on the original image and then the other half on the cross batch exampel
                    img_comb  = [self.transform[0](img)]                 
                    img_comb += [self.transform[1](cb_img)]
                    img_comb += [tr(img)    for tr in self.transform[2:5]]
                    img_comb += [tr(cb_img) for tr in self.transform[5:]]
                    img = img_comb
                    
                else:
                    if self.is_multi_crop:
                        img = self.multi_crop_aug(img, self.transform)
                    else:
                        img = [self.transform(img) for _ in range(self.num_augmentations)]
                img = img[0] if len(img) == 1 and isinstance(img, list) else img      

            return img, label
            
            
        else:
            if self.resizing is not None:
                img = self.resizing(img)

            if self.transform is not None:
                if isinstance(self.transform, list):
                    img = [tr(img) for tr in self.transform]
                else:
                    if self.is_multi_crop:
                        img = self.multi_crop_aug(img, self.transform)
                    else:
                        img = [self.transform(img) for _ in range(self.num_augmentations)]
                img = img[0] if len(img) == 1 and isinstance(img, list) else img      

            return img, label
