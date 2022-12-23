import torch
import torch.nn as nn
from focal_loss import FocalLoss
import distance

class MMFN(nn.Module):
    def __init__(self, model_param):
        super(MMFN, self).__init__()
        if 'video' in model_param:
            video_model = model_param["video"]["model"]
            for param in video_model.parameters():
                param.requires_grad = False
            for param in video_model.fc6.parameters():
                param.requires_grad = True
            for param in video_model.fc7.parameters():
                param.requires_grad = True
            for param in video_model.fc8.parameters():
                param.requires_grad = True
            self.video_model_blocks = video_model
            self.video_id = model_param["video"]["id"]
        if "touch" in model_param:
            touch_model = model_param["touch"]["model"]
            self.touch_model_blocks = touch_model
            self.touch_id = model_param["touch"]["id"]
        self.loss_func = nn.CrossEntropyLoss()
        self.loss_focal = FocalLoss()
        self.loss_diff = DiffLoss()
        self.loss_smi = distance.CMD()
        self.vedio_fc_share = nn.Linear(4096, 64)
        self.touch_fc_share = nn.Linear(256,64)
        self.vedio_fc_spe = nn.Linear(4096, 64)
        self.touch_fc_spe = nn.Linear(256, 64)
        self.vedio_fc_pre = nn.Linear(64, 6)
        self.fc1 = nn.Linear(128, 6)

        self.mmtm = MMTM(64, 64, 4)

    def forward(self, ges, face, y):
        ############################################## FACE BLOCK1
        x_f = self.video_model_blocks.relu(self.video_model_blocks.conv1(face))
        x_f = self.video_model_blocks.pool1(x_f)

        x_f = self.video_model_blocks.relu(self.video_model_blocks.conv2(x_f))
        x_f = self.video_model_blocks.pool2(x_f)

        x_f = self.video_model_blocks.relu(self.video_model_blocks.conv3a(x_f))
        x_f = self.video_model_blocks.relu(self.video_model_blocks.conv3b(x_f))
        x_f = self.video_model_blocks.pool3(x_f)

        x_f = self.video_model_blocks.relu(self.video_model_blocks.conv4a(x_f))
        x_f = self.video_model_blocks.relu(self.video_model_blocks.conv4b(x_f))
        x_f = self.video_model_blocks.pool4(x_f)

        ############################################## TOUCH BLOCK1
        x_t = self.touch_model_blocks.layer1(ges)
        # x = F.dropout(x, 0.25)
        x_t = self.touch_model_blocks.max1(x_t)

        x_t = self.touch_model_blocks.layer2(x_t)
        # x = F.dropout(x, 0.25)
        x_t = self.touch_model_blocks.max2(x_t)

        x_t = self.touch_model_blocks.layer3(x_t)
        # x = F.dropout(x, 0.25)
        x_t = self.touch_model_blocks.max3(x_t)
        #################################### FIRST MMTM1


        ############################################## FACE BLOCK2
        x_f = self.video_model_blocks.relu(self.video_model_blocks.conv5a(x_f))
        x_f = self.video_model_blocks.relu(self.video_model_blocks.conv5b(x_f))
        x_f = self.video_model_blocks.pool5(x_f)
        ############################################## TOUCH BLOCK2
        x_t = self.touch_model_blocks.layer4(x_t)
        # x = F.dropout(x, 0.25)
        x_t = self.touch_model_blocks.max4(x_t)
        #################################### Second MMTM2


        ############################################## FACE BLOCK3
        x_f = x_f.view(-1, 8192 * 4)  # 8192
        x_f = self.video_model_blocks.relu(self.video_model_blocks.fc6(x_f))
        x_f = self.video_model_blocks.dropout(x_f)
        x_f = self.video_model_blocks.relu(self.video_model_blocks.fc7(x_f))
        x_f = self.video_model_blocks.dropout(x_f)  #4096
        x_f_share = self.vedio_fc_share(x_f)  #64
        x_f_specefic = self.vedio_fc_spe(x_f)  #64
        face_pre = self.vedio_fc_pre(x_f_specefic)  #6
        ############################################## TOUCH BLOCK3
        x_t = self.touch_model_blocks.avgpool(x_t)
        # Flatten the layer to fc
        x_t = x_t.flatten(1)  #256
        x_t_specefic = self.touch_model_blocks.fc2(x_t)  #64
        x_t_share = self.touch_fc_share(x_t)  #64
        ges_pre = self.touch_model_blocks.fc3(x_t_specefic)

        #################################### Second MMTM2
        x_f_1, x_t_1 = self.mmtm(x_f_share, x_t_share)
        ####################################
        x_f_share = x_f_1 + x_f_share
        x_t_share = x_t_1 + x_t_share

        y_fusion = torch.cat((x_f_share,x_t_share),1)
        share_pre = self.fc1(y_fusion)

        fusion_pre = face_pre + ges_pre + share_pre

        loss = self.loss_func(face_pre, y) + self.loss_func(ges_pre, y) + self.loss_func(share_pre, y)
        loss_smi = ((x_f_share - x_t_share) ** 2).sum(1).sqrt().mean()
        loss_diff = self.loss_diff(x_f_specefic, x_f_share) + self.loss_diff(x_t_specefic, x_t_share)
        return face_pre, ges_pre, fusion_pre, loss + 0.001 * loss_smi + 0.001 * loss_diff



class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

def init_weights(m):
 print(m)
 if type(m) == nn.Linear:
   print(m.weight)
 else:
   print('error')

class MMTM(nn.Module):
  def __init__(self, dim_visual, dim_skeleton, ratio):
    super(MMTM, self).__init__()
    dim = dim_visual + dim_skeleton
    dim_out = int(2*dim/ratio)
    self.fc_squeeze = nn.Linear(dim, dim_out)

    self.fc_visual = nn.Linear(dim_out, dim_visual)
    self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

    # initialize
    with torch.no_grad():
      self.fc_squeeze.apply(init_weights)
      self.fc_visual.apply(init_weights)
      self.fc_skeleton.apply(init_weights)

  def forward(self, visual, skeleton):
    squeeze_array = []
    for tensor in [visual, skeleton]:
      tview = tensor.view(tensor.shape[:2] + (-1,))
      squeeze_array.append(torch.mean(tview, dim=-1))
    squeeze = torch.cat(squeeze_array, 1)

    excitation = self.fc_squeeze(squeeze)
    excitation = self.relu(excitation)

    vis_out = self.fc_visual(excitation)
    sk_out = self.fc_skeleton(excitation)

    vis_out = self.sigmoid(vis_out)
    sk_out = self.sigmoid(sk_out)

    dim_diff = len(visual.shape) - len(vis_out.shape)
    vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

    dim_diff = len(skeleton.shape) - len(sk_out.shape)
    sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

    return visual * vis_out, skeleton * sk_out