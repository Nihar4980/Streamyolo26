from block import *
class yolon(nn.Module):
    def __init__(self,
                 scale="nano",
                 nc=8, 
                 reg_max=1,
                 global_attn = False,
                 improver = False ,
                ):
        super().__init__()

        cfg = {
            "nano":  dict(w=0.25),
        }[scale]

        def c(ch):  # channel scaler
            return int(ch * cfg["w"])
        self.scale = scale                                                                
        self.global_attn = global_attn
        self.improver = improver

        # CNN stem
        self.conv1 = Conv(3,c(64), k=3, s=2, p=1)
        self.conv2 = Conv(c(64),c(128), k=3, s=2, p=1)
        self.c3_1 = C3k2(c1=c(128),c2=c(256),e=.25, n=1)
        self.conv3 = Conv(c(256),c(256), 3, s=2, p=1)
        self.c3_2 = C3k2(c1=c(256),c2=c(512),e= .25,n=1) 

        
         # Transformer branch
        if self.global_attn:
            self.na = NAVisionTransformer(embed_dim=c(512),num_heads=4,patch_size=8,depth = 4)
            self.merge_conv = Conv(2*c(512), c(512), 1)
        self.conv4 = Conv(c(512),c(512), k=3, s=2, p=1)
        self.c3_3 = C3k2(c1=c(512),c2=c(512),c3k= True,n=1)  

        self.conv5 = Conv(c(512),c(1024), 3, s=2, p=1)
        self.c3_4 = C3k2(c1=c(1024),c2=c(1024),c3k= True, n=1)
        #  # Transformer branch
        # if self.ace:
        #     self.ace = HyperACE(
        #                             c1=c(512),
        #                             c2=c(512),
        #                             n=2,
        #                        )
        #     self.fullpad = FullPAD_Tunnel()
        # else:
        self.sppf = SPPF( c(1024),c(1024),5, 3, True)
        self.c2psa = C2PSA(c(1024),c(1024))
        

        # Neck: Feature aggregation and pyramid creation
        
        #self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        #self.reduce1 = Conv(c(1024) + c(512), c(512), 1)
        self.fuse1 = C3k2(c1=c(1024+512),c2=c(512),c3k= True,n=1) 

        #self.upsample2 = nn.Upsample(size=(75, 120), mode='nearest')
        #self.reduce2 = Conv(c(512) + c(256), c(256), k=1)
        self.fuse2 = C3k2(c1=c(512+512),c2=c(256),c3k= True,n=1) 

        # Bottom-up pathway
        self.down1 = Conv(c(256), c(256), k=3, s=2, p=1,)
        #self.merge1 = Conv(c(512) + c(512), c(512), k=1)
        self.fuse3 = C3k2(c1=c(256+512),c2=c(512),c3k= True,n=1) 

        self.down2 = Conv(c(512), c(512), k=3, s=2, p=1, )
        #self.merge2 = Conv(c(1024) + c(1024), c(1024), k=1)
        self.fuse4 = C3k2(c1=c(512+1024),c2=c(1024),attn=True, n=1)

        # =========================
        # Temporal Fusion (SCA)
        # =========================
        
        self.join0 = Conv(c(256), c(256)//2, 1)
        self.join1 = Conv(c(512), c(512)//2, 1)
        self.join2 = Conv(c(1024), c(1024)//2, 1)

        self.improver0 = C3k2(c1=c(256),c2=c(256),c3k= True,n=1) 
        self.improver1 = C3k2(c1=c(512),c2=c(512),c3k= True,n=1)
        self.improver2 = C3k2(c1=c(1024),c2=c(1024),c3k= True, n=1)
        
        # Initialize only the improver modules
        self._init_weights(self.join0)
        self._init_weights(self.join1)
        self._init_weights(self.join2)
        if self.improver:
            self._init_weights(self.improver0)
            self._init_weights(self.improver1)
            self._init_weights(self.improver2)
        if self.global_attn:
            self._init_weights(self.na)
            self._init_weights(self.merge_conv)

        # =========================
        # Detect Head
        # =========================
        self.detect = Detect(nc=nc, reg_max = reg_max, ch=(c(256), c(512), c(1024)))

    def _init_weights(self, module):
        """Initialize model weights to random values."""
        for m in module.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.1
                
            elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU,nn.GELU}:
                m.inplace = True



    def off_forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        current = torch.split(input, 3, dim=1)[0]
        current_x1 = self.conv1(current)
        current_x2 = self.c3_1(self.conv2(current_x1))
        current_x3 = self.c3_2(self.conv3(current_x2))
        if self.global_attn:
            current_t = self.na(current)
            current_x3 = self.merge_conv(torch.cat([current_x3,current_t], 1))
        current_x4 = self.c3_3(self.conv4(current_x3))
        current_x5 = self.c3_4(self.conv5(current_x4))
        current_x5 = self.c2psa(self.sppf(current_x5))
        current_x6 = self.fuse1(torch.cat([F.interpolate(current_x5, size=current_x4.shape[2:4], mode='nearest'), current_x4], 1))
        current_p0 = self.fuse2(torch.cat([F.interpolate(current_x6, size=current_x3.shape[2:4], mode='nearest'), current_x3], 1))
        current_p1 = self.fuse3(torch.cat([self.down1(current_p0),current_x6], 1))
        current_p2 = self.fuse4(torch.cat([self.down2(current_p1), current_x5], 1))

        support = torch.split(input, 3, dim=1)[1]
        support_x1 = self.conv1(support)
        support_x2 = self.c3_1(self.conv2(support_x1))
        support_x3 = self.c3_2(self.conv3(support_x2))
        if self.global_attn:
            support_t = self.na(support)
            support_x3 = self.merge_conv(torch.cat([support_x3,support_t], 1))
        support_x4 = self.c3_3(self.conv4(support_x3))
        support_x5 = self.c3_4(self.conv5(support_x4))
        support_x5 = self.c2psa(self.sppf(support_x5))
        support_x6 = self.fuse1(torch.cat([F.interpolate(support_x5, size=support_x4.shape[2:4], mode='nearest'), support_x4], 1))
        support_p0 = self.fuse2(torch.cat([F.interpolate(support_x6, size=support_x3.shape[2:4], mode='nearest'), support_x3], 1))
        support_p1 = self.fuse3(torch.cat([self.down1(support_p0),support_x6], 1))
        support_p2 = self.fuse4(torch.cat([self.down2(support_p1), support_x5], 1))
        
        if self.improver:
            p0 = self.improver0(torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1)) + current_p0
            p1 = self.improver1(torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1)) + current_p1
            p2 = self.improver2(torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1)) + current_p2
        else:
            p0 = torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1) + current_p0
            p1 = torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1) + current_p1
            p2 = torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1) + current_p2
            
        outputs = [p0,p1,p2]

        return outputs

    def online_forward(self, input, buffer=None, node='star'):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        current_x1 = self.conv1(input)
        current_x2 = self.c3_1(self.conv2(current_x1))
        current_x3 = self.c3_2(self.conv3(current_x2))
        if self.global_attn:
            current_t = self.na(input)
            current_x3 = self.merge_conv(torch.cat([current_x3,current_t], 1))
        current_x4 = self.c3_3(self.conv4(current_x3))
        current_x5 = self.c3_4(self.conv5(current_x4))
        current_x5 = self.c2psa(self.sppf(current_x5))
        current_x6 = self.fuse1(torch.cat([F.interpolate(current_x5, size=current_x4.shape[2:4], mode='nearest'), current_x4], 1))
        current_p0 = self.fuse2(torch.cat([F.interpolate(current_x6, size=current_x3.shape[2:4], mode='nearest'), current_x3], 1))
        current_p1 = self.fuse3(torch.cat([self.down1(current_p0),current_x6], 1))
        current_p2 = self.fuse4(torch.cat([self.down2(current_p1), current_x5], 1))
        
        #####
        if node=='star':
            if self.improver:
                p0 = self.improver0(torch.cat([self.join0(current_p0), self.join0(current_p0)], dim=1)) + current_p0
                p1 = self.improver1(torch.cat([self.join1(current_p1), self.join1(current_p1)], dim=1)) + current_p1
                p2 = self.improver2(torch.cat([self.join2(current_p2), self.join2(current_p2)], dim=1)) + current_p2
            else:
                p0 = torch.cat([self.join0(current_p0), self.join0(current_p0)], dim=1) + current_p0
                p1 = torch.cat([self.join1(current_p1), self.join1(current_p1)], dim=1) + current_p1
                p2 = torch.cat([self.join2(current_p2), self.join2(current_p2)], dim=1) + current_p2

        elif node=='buffer':
            [support_p0, support_p1, support_p2] = buffer
            
            if self.improver:
                p0 = self.improver0(torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1)) + current_p0
                p1 = self.improver1(torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1)) + current_p1
                p2 = self.improver2(torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1)) + current_p2
            else:
                p0 = torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1) + current_p0
                p1 = torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1) + current_p1
                p2 = torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1) + current_p2

        outputs = [p0,p1,p2]
        buffer_ = [current_p0,current_p1,current_p2]

        return outputs, buffer_

    def forward(self, input, buffer=None, mode='off_pipe'):

        if mode=='off_pipe':
            # Glops caculate mode
            if input.size()[1] == 3:
                input = torch.cat([input, input], dim=1)
                output = self.off_forward(input)
            # offline train mode
            elif input.size()[1] == 6:
                output = self.off_forward(input)
            
            return self.detect(output)
        
        elif mode=='on_pipe':
            # online star state
            if buffer == None:
                output, buffer_ = self.online_forward(input, node='star')
            # online inference
            else:
                assert len(buffer) == 3
                assert input.size()[1] == 3
                output, buffer_ = self.online_forward(input, buffer=buffer, node='buffer')
            
            return self.detect(output), buffer_ 
class yolos(nn.Module):
    def __init__(self,
                 scale="small",
                 nc=8, 
                 reg_max=16,
                 global_attn = False,
                 improver = False ,
                ):
        super().__init__()

        cfg = {
            "small":  dict(w=0.50),
        }[scale]

        def c(ch):  # channel scaler
            return int(ch * cfg["w"])
        self.scale = scale                                                                
        self.global_attn = global_attn
        self.improver = improver

        # CNN stem
        self.conv1 = Conv(3,c(64), k=3, s=2, p=1)
        self.conv2 = Conv(c(64),c(128), k=3, s=2, p=1)
        self.c3_1 = C3k2(c1=c(128),c2=c(256),e=.25, n=1)
        self.conv3 = Conv(c(256),c(256), 3, s=2, p=1)
        self.c3_2 = C3k2(c1=c(256),c2=c(512),e= .25,n=1) 

        
         # Transformer branch
        if self.global_attn:
            self.na = NAVisionTransformer(embed_dim=c(512),num_heads=4,patch_size=8,depth = 4)
            self.merge_conv = Conv(2*c(512), c(512), 1)
        self.conv4 = Conv(c(512),c(512), k=3, s=2, p=1)
        self.c3_3 = C3k2(c1=c(512),c2=c(512),c3k= True,n=1)  

        self.conv5 = Conv(c(512),c(1024), 3, s=2, p=1)
        self.c3_4 = C3k2(c1=c(1024),c2=c(1024),c3k= True, n=1)
        #  # Transformer branch
        # if self.ace:
        #     self.ace = HyperACE(
        #                             c1=c(512),
        #                             c2=c(512),
        #                             n=2,
        #                        )
        #     self.fullpad = FullPAD_Tunnel()
        # else:
        self.sppf = SPPF( c(1024),c(1024),5, 3, True)
        self.c2psa = C2PSA(c(1024),c(1024))
        

        # Neck: Feature aggregation and pyramid creation
        
        #self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        #self.reduce1 = Conv(c(1024) + c(512), c(512), 1)
        self.fuse1 = C3k2(c1=c(1024+512),c2=c(512),c3k= True,n=1) 

        #self.upsample2 = nn.Upsample(size=(75, 120), mode='nearest')
        #self.reduce2 = Conv(c(512) + c(256), c(256), k=1)
        self.fuse2 = C3k2(c1=c(512+512),c2=c(256),c3k= True,n=1) 

        # Bottom-up pathway
        self.down1 = Conv(c(256), c(256), k=3, s=2, p=1,)
        #self.merge1 = Conv(c(512) + c(512), c(512), k=1)
        self.fuse3 = C3k2(c1=c(256+512),c2=c(512),c3k= True,n=1) 

        self.down2 = Conv(c(512), c(512), k=3, s=2, p=1, )
        #self.merge2 = Conv(c(1024) + c(1024), c(1024), k=1)
        self.fuse4 = C3k2(c1=c(512+1024),c2=c(1024),attn=True, n=1)

        # =========================
        # Temporal Fusion (SCA)
        # =========================
        
        self.join0 = Conv(c(256), c(256)//2, 1)
        self.join1 = Conv(c(512), c(512)//2, 1)
        self.join2 = Conv(c(1024), c(1024)//2, 1)

        self.improver0 = C3k2(c1=c(256),c2=c(256),c3k= True,n=1) 
        self.improver1 = C3k2(c1=c(512),c2=c(512),c3k= True,n=1)
        self.improver2 = C3k2(c1=c(1024),c2=c(1024),c3k= True, n=1)
        
        # Initialize only the improver modules
        self._init_weights(self.join0)
        self._init_weights(self.join1)
        self._init_weights(self.join2)
        if self.improver:
            self._init_weights(self.improver0)
            self._init_weights(self.improver1)
            self._init_weights(self.improver2)
        if self.global_attn:
            self._init_weights(self.na)
            self._init_weights(self.merge_conv)

        # # Crossattention
        # if self.crossattention:
        #     self.sca0 = SCABlock(dim=c(256), num_heads=4,kernel_size=(3,5))
        #     self.sca1 = SCABlock(dim=c(512), num_heads=4,kernel_size=(5,7))
        #     self.sca2 = SCABlock(dim=c(1024), num_heads=4,kernel_size=(7,9))
        #     self._init_weights(self.sca0)
        #     self._init_weights(self.sca1)
        #     self._init_weights(self.sca2)
        


        # self.spatial_imp_1 = ImplicitShift_S(round(self.h/8), round(self.w/8), num_imp=self.num_imp_s, gram_mode=self.gram_mode,scale=self.scale)
        # self.spatial_imp_2 = ImplicitShift_S(round(self.h/16), round(self.w/16), num_imp=self.num_imp_s, gram_mode=self.gram_mode,scale=self.scale)
        # self.spatial_imp_3 = ImplicitShift_S(round(self.h/32), round(self.w/32), num_imp=self.num_imp_s, gram_mode=self.gram_mode,scale=self.scale)
            

        # =========================
        # Detect Head
        # =========================
        self.detect = Detect(nc=nc, reg_max = reg_max, ch=(c(256), c(512), c(1024)))

    def _init_weights(self, module):
        """Initialize model weights to random values."""
        for m in module.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.1
                
            elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU,nn.GELU}:
                m.inplace = True



    def off_forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        current = torch.split(input, 3, dim=1)[0]
        current_x1 = self.conv1(current)
        current_x2 = self.c3_1(self.conv2(current_x1))
        current_x3 = self.c3_2(self.conv3(current_x2))
        if self.global_attn:
            current_t = self.na(current)
            current_x3 = self.merge_conv(torch.cat([current_x3,current_t], 1))
        current_x4 = self.c3_3(self.conv4(current_x3))
        current_x5 = self.c3_4(self.conv5(current_x4))
        current_x5 = self.c2psa(self.sppf(current_x5))
        current_x6 = self.fuse1(torch.cat([F.interpolate(current_x5, size=current_x4.shape[2:4], mode='nearest'), current_x4], 1))
        current_p0 = self.fuse2(torch.cat([F.interpolate(current_x6, size=current_x3.shape[2:4], mode='nearest'), current_x3], 1))
        current_p1 = self.fuse3(torch.cat([self.down1(current_p0),current_x6], 1))
        current_p2 = self.fuse4(torch.cat([self.down2(current_p1), current_x5], 1))

        support = torch.split(input, 3, dim=1)[1]
        support_x1 = self.conv1(support)
        support_x2 = self.c3_1(self.conv2(support_x1))
        support_x3 = self.c3_2(self.conv3(support_x2))
        if self.global_attn:
            support_t = self.na(support)
            support_x3 = self.merge_conv(torch.cat([support_x3,support_t], 1))
        support_x4 = self.c3_3(self.conv4(support_x3))
        support_x5 = self.c3_4(self.conv5(support_x4))
        support_x5 = self.c2psa(self.sppf(support_x5))
        support_x6 = self.fuse1(torch.cat([F.interpolate(support_x5, size=support_x4.shape[2:4], mode='nearest'), support_x4], 1))
        support_p0 = self.fuse2(torch.cat([F.interpolate(support_x6, size=support_x3.shape[2:4], mode='nearest'), support_x3], 1))
        support_p1 = self.fuse3(torch.cat([self.down1(support_p0),support_x6], 1))
        support_p2 = self.fuse4(torch.cat([self.down2(support_p1), support_x5], 1))
        
        if self.improver:
            p0 = self.improver0(torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1)) + current_p0
            p1 = self.improver1(torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1)) + current_p1
            p2 = self.improver2(torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1)) + current_p2
        else:
            p0 = torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1) + current_p0
            p1 = torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1) + current_p1
            p2 = torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1) + current_p2
            
        outputs = [p0,p1,p2]

        return outputs

    def online_forward(self, input, buffer=None, node='star'):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        current_x1 = self.conv1(input)
        current_x2 = self.c3_1(self.conv2(current_x1))
        current_x3 = self.c3_2(self.conv3(current_x2))
        if self.global_attn:
            current_t = self.na(input)
            current_x3 = self.merge_conv(torch.cat([current_x3,current_t], 1))
        current_x4 = self.c3_3(self.conv4(current_x3))
        current_x5 = self.c3_4(self.conv5(current_x4))
        current_x5 = self.c2psa(self.sppf(current_x5))
        current_x6 = self.fuse1(torch.cat([F.interpolate(current_x5, size=current_x4.shape[2:4], mode='nearest'), current_x4], 1))
        current_p0 = self.fuse2(torch.cat([F.interpolate(current_x6, size=current_x3.shape[2:4], mode='nearest'), current_x3], 1))
        current_p1 = self.fuse3(torch.cat([self.down1(current_p0),current_x6], 1))
        current_p2 = self.fuse4(torch.cat([self.down2(current_p1), current_x5], 1))
        
        #####
        if node=='star':
            if self.improver:
                p0 = self.improver0(torch.cat([self.join0(current_p0), self.join0(current_p0)], dim=1)) + current_p0
                p1 = self.improver1(torch.cat([self.join1(current_p1), self.join1(current_p1)], dim=1)) + current_p1
                p2 = self.improver2(torch.cat([self.join2(current_p2), self.join2(current_p2)], dim=1)) + current_p2
            else:
                p0 = torch.cat([self.join0(current_p0), self.join0(current_p0)], dim=1) + current_p0
                p1 = torch.cat([self.join1(current_p1), self.join1(current_p1)], dim=1) + current_p1
                p2 = torch.cat([self.join2(current_p2), self.join2(current_p2)], dim=1) + current_p2

        elif node=='buffer':
            [support_p0, support_p1, support_p2] = buffer
            
            if self.improver:
                p0 = self.improver0(torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1)) + current_p0
                p1 = self.improver1(torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1)) + current_p1
                p2 = self.improver2(torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1)) + current_p2
            else:
                p0 = torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1) + current_p0
                p1 = torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1) + current_p1
                p2 = torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1) + current_p2

        outputs = [p0,p1,p2]
        buffer_ = [current_p0,current_p1,current_p2]

        return outputs, buffer_

    def forward(self, input, buffer=None, mode='off_pipe'):

        if mode=='off_pipe':
            # Glops caculate mode
            if input.size()[1] == 3:
                input = torch.cat([input, input], dim=1)
                output = self.off_forward(input)
            # offline train mode
            elif input.size()[1] == 6:
                output = self.off_forward(input)
            
            return self.detect(output)
        
        elif mode=='on_pipe':
            # online star state
            if buffer == None:
                output, buffer_ = self.online_forward(input, node='star')
            # online inference
            else:
                assert len(buffer) == 3
                assert input.size()[1] == 3
                output, buffer_ = self.online_forward(input, buffer=buffer, node='buffer')
            
            return self.detect(output), buffer_ 

        
            
              


class yolom(nn.Module):
    def __init__(self,
                 scale="small",
                 nc=8, 
                 reg_max=16,
                 gram_mode="gram",
                 streamyolo = False,
                 spatial_imp = False,       
                 h = 600,                                                                      
                 w = 960,                                                                             
                 num_imp_s=1,
                 ace =False,
                 global_attn = False,
                 improver = False):
        super().__init__()

        cfg = {
            "small":  dict(w=0.50),
            "medium": dict(w=1),
            "large":  dict(w=1.00),
        }[scale]

        def c(ch):  # channel scaler
            return int(ch * cfg["w"])
        self.scale = scale                                                                
        self.gram_mode = gram_mode   
        self.spatial_imp = spatial_imp  
        self.streamyolo = streamyolo
        self.num_imp_s = num_imp_s                                                      
        self.h = h 
        self.w = w 
        self.ace = ace
        self.global_attn = global_attn
        self.improver = improver


        # CNN stem
        self.conv1 = Conv(3,64, k=3, s=2, p=1)
        self.conv2 = Conv(64,128, k=3, s=2, p=1)
        self.c3_1 = C3k2(128,256,e=0.25, c3k=True)
        self.conv3 = Conv(256,256, 3, s=2, p=1)
        self.c3_2 = C3k2(c1=256,c2=512,e=0.25,c3k= True) 

        
         # Transformer branch
        if self.global_attn:
            self.na = NAVisionTransformer(embed_dim=512,num_heads=4,patch_size=8,flatten_embedding=True,depth = 2)
            self.merge_conv = Conv(2*512, 512, 1)
        self.conv4 = Conv(512,512, k=3, s=2, p=1)
        self.c3_3 = C3k2(c1=512,c2=512,c3k= True,n=1)  

        self.conv5 = Conv(512,512, 3, s=2, p=1)
        self.c3_4 = C3k2(512,512,c3k= True, n=1)
        #  # Transformer branch
        # if self.ace:
        #     self.ace = HyperACE(
        #                             c1=c(512),
        #                             c2=c(512),
        #                             n=2,
        #                        )
        #     self.fullpad = FullPAD_Tunnel()
        # else:
        self.sppf = SPPF( 512,512,5, 3, True)
        self.c2psa = C2PSA(512,512)
            

        # Neck: Feature aggregation and pyramid creation
        
        #self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        #self.reduce1 = Conv(c(1024) + c(512), c(512), 1)
        self.fuse1 = C3k2(c1=1024,c2=512,c3k= True) 

        #self.upsample2 = nn.Upsample(size=(75, 120), mode='nearest')
        #self.reduce2 = Conv(c(512) + c(256), c(256), k=1)
        self.fuse2 = C3k2(c1=1024,c2=256,c3k= True,n=1) 

        # Bottom-up pathway
        self.down1 = Conv(256, 256, k=3, s=2, p=1,)
        #self.merge1 = Conv(c(512) + c(512), c(512), k=1)
        self.fuse3 = C3k2(c1=768,c2=512,e=0.5,c3k= True,n=1)

        self.down2 = Conv(512, 512, k=3, s=2, p=1,)
        #self.merge2 = Conv(c(1024) + c(1024), c(1024), k=1)
        self.fuse4 = C3k2(c1=1024,c2=512,attn=True, n=1)

        # =========================
        # Temporal Fusion (SCA)
        # =========================
        self.join0 = Conv(256, 256//2, 1)
        self.join1 = Conv(512, 512//2, 1)
        self.join2 = Conv(512, 512//2, 1)

        self.improver0 = C3k2(c1=256,c2=256,c3k=True,n=1) 
        self.improver1  = C3k2(c1=512,c2=512,c3k=True,n=1)
        self.improver2  = C3k2(c1=512,c2=512,c3k=True, n=1)
        # Initialize only the improver modules
        # Initialize only the improver modules
        self._init_weights(self.join0)
        self._init_weights(self.join1)
        self._init_weights(self.join2)
        if self.improver:
            self._init_weights(self.improver0)
            self._init_weights(self.improver1)
            self._init_weights(self.improver2)
        if self.global_attn:
            self._init_weights(self.na)
            self._init_weights(self.merge_conv)


        # self.spatial_imp_1 = ImplicitShift_S(round(self.h/8), round(self.w/8), num_imp=self.num_imp_s, gram_mode=self.gram_mode,scale=self.scale)
        # self.spatial_imp_2 = ImplicitShift_S(round(self.h/16), round(self.w/16), num_imp=self.num_imp_s, gram_mode=self.gram_mode,scale=self.scale)
        # self.spatial_imp_3 = ImplicitShift_S(round(self.h/32), round(self.w/32), num_imp=self.num_imp_s, gram_mode=self.gram_mode,scale=self.scale)
            

        # =========================
        # Detect Head
        # =========================
        self.detect = Detect(nc=nc, reg_max = reg_max, ch=(256,512, 512))
        
    def _init_weights(self, module):
        """Initialize model weights to random values."""
        for m in module.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.1
                
            elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU,nn.GELU}:
                m.inplace = True

    # -------------------------------------------------
    # Single-frame backbone forward
    # -------------------------------------------------
    def off_forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        current = torch.split(input, 3, dim=1)[0]
        current_x1 = self.conv1(current)
        current_x2 = self.c3_1(self.conv2(current_x1))
        current_x3 = self.c3_2(self.conv3(current_x2))
        if self.global_attn:
            current_t = self.na(current)
            current_x3 = self.merge_conv(torch.cat([current_x3,current_t], 1))
        current_x4 = self.c3_3(self.conv4(current_x3))
        current_x5 = self.c3_4(self.conv5(current_x4))
        current_x5 = self.c2psa(self.sppf(current_x5))
        current_x6 = self.fuse1(torch.cat([F.interpolate(current_x5, size=current_x4.shape[2:4], mode='nearest'), current_x4], 1))
        current_p0 = self.fuse2(torch.cat([F.interpolate(current_x6, size=current_x3.shape[2:4], mode='nearest'), current_x3], 1))
        current_p1 = self.fuse3(torch.cat([self.down1(current_p0),current_x6], 1))
        current_p2 = self.fuse4(torch.cat([self.down2(current_p1), current_x5], 1))

        support = torch.split(input, 3, dim=1)[1]
        support_x1 = self.conv1(support)
        support_x2 = self.c3_1(self.conv2(support_x1))
        support_x3 = self.c3_2(self.conv3(support_x2))
        if self.global_attn:
            support_t = self.na(support)
            support_x3 = self.merge_conv(torch.cat([support_x3,support_t], 1))
        support_x4 = self.c3_3(self.conv4(support_x3))
        support_x5 = self.c3_4(self.conv5(support_x4))
        support_x5 = self.c2psa(self.sppf(support_x5))
        support_x6 = self.fuse1(torch.cat([F.interpolate(support_x5, size=support_x4.shape[2:4], mode='nearest'), support_x4], 1))
        support_p0 = self.fuse2(torch.cat([F.interpolate(support_x6, size=support_x3.shape[2:4], mode='nearest'), support_x3], 1))
        support_p1 = self.fuse3(torch.cat([self.down1(support_p0),support_x6], 1))
        support_p2 = self.fuse4(torch.cat([self.down2(support_p1), support_x5], 1))
        if self.improver:
            p0 = self.improver0(torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1)) + current_p0
            p1 = self.improver1(torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1)) + current_p1
            p2 = self.improver2(torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1)) + current_p2
        else:
            p0 = torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1) + current_p0
            p1 = torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1) + current_p1
            p2 = torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1) + current_p2
            
        outputs = [p0,p1,p2]

        return outputs

    def online_forward(self, input, buffer=None, node='star'):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        current_x1 = self.conv1(input)
        current_x2 = self.c3_1(self.conv2(current_x1))
        current_x3 = self.c3_2(self.conv3(current_x2))
        if self.global_attn:
            current_t = self.na(input)
            current_x3 = self.merge_conv(torch.cat([current_x3,current_t], 1))
        current_x4 = self.c3_3(self.conv4(current_x3))
        current_x5 = self.c3_4(self.conv5(current_x4))
        current_x5 = self.c2psa(self.sppf(current_x5))
        current_x6 = self.fuse1(torch.cat([F.interpolate(current_x5, size=current_x4.shape[2:4], mode='nearest'), current_x4], 1))
        current_p0 = self.fuse2(torch.cat([F.interpolate(current_x6, size=current_x3.shape[2:4], mode='nearest'), current_x3], 1))
        current_p1 = self.fuse3(torch.cat([self.down1(current_p0),current_x6], 1))
        current_p2 = self.fuse4(torch.cat([self.down2(current_p1), current_x5], 1))
        
        #####
        if node=='star':
            if self.improver:
                p0 = self.improver0(torch.cat([self.join0(current_p0), self.join0(current_p0)], dim=1)) + current_p0
                p1 = self.improver1(torch.cat([self.join1(current_p1), self.join1(current_p1)], dim=1)) + current_p1
                p2 = self.improver2(torch.cat([self.join2(current_p2), self.join2(current_p2)], dim=1)) + current_p2
            else:
                p0 = torch.cat([self.join0(current_p0), self.join0(current_p0)], dim=1) + current_p0
                p1 = torch.cat([self.join1(current_p1), self.join1(current_p1)], dim=1) + current_p1
                p2 = torch.cat([self.join2(current_p2), self.join2(current_p2)], dim=1) + current_p2

        elif node=='buffer':
            [support_p0, support_p1, support_p2] = buffer

            if self.improver:
                p0 = self.improver0(torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1)) + current_p0
                p1 = self.improver1(torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1)) + current_p1
                p2 = self.improver2(torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1)) + current_p2
            else:
                p0 = torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1) + current_p0
                p1 = torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1) + current_p1
                p2 = torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1) + current_p2

        outputs = [p0,p1,p2]
        buffer_ = [current_p0,current_p1,current_p2]

        return outputs, buffer_

    def forward(self, input, buffer=None, mode='off_pipe'):

        if mode=='off_pipe':
            # Glops caculate mode
            if input.size()[1] == 3:
                input = torch.cat([input, input], dim=1)
                output = self.off_forward(input)
            # offline train mode
            elif input.size()[1] == 6:
                output = self.off_forward(input)
            
            return self.detect(output)
        
        elif mode=='on_pipe':
            # online star state
            if buffer == None:
                output, buffer_ = self.online_forward(input, node='star')
            # online inference
            else:
                assert len(buffer) == 3
                assert input.size()[1] == 3
                output, buffer_ = self.online_forward(input, buffer=buffer, node='buffer')
            
            return self.detect(output), buffer_ 


class yolol(nn.Module):
    def __init__(self,
                 nc=8, 
                 reg_max=16,
                 global_attn = False,
                 improver = False):
                 #scale="small",
                 
                 # #gram_mode="gram",
                 # #streamyolo = False,
                 # spatial_imp = False,       
                 # h = 600,                                                                      
                 # w = 960,                                                                             
                 # num_imp_s=1,
                 
        super().__init__()

        # cfg = {
        #     "small":  dict(w=0.50),
        #     "medium": dict(w=1),
        #     "large":  dict(w=1.00),
        # }[scale]

        # def c(ch):  # channel scaler
        #     return int(ch * cfg["w"])
        # self.scale = scale                                                                
        # self.gram_mode = gram_mode   
        # self.spatial_imp = spatial_imp  
        # self.streamyolo = streamyolo
        # self.num_imp_s = num_imp_s                                                      
        # self.h = h 
        # self.w = w 
        self.global_attn = global_attn
        self.improver = improver


        # CNN stem
        self.conv1 = Conv(3,64, k=3, s=2, p=1)
        self.conv2 = Conv(64,128, k=3, s=2, p=1)
        self.c3_1 = C3k2(128,256,e=0.25, c3k=True, n=2)
        self.conv3 = Conv(256,256, 3, s=2, p=1)
        self.c3_2 = C3k2(c1=256,c2=512,e=0.25,c3k= True, n=2) 

        
         # Transformer branch
        if self.global_attn:
            self.na = NAVisionTransformer(embed_dim=512,num_heads=4,patch_size=8,flatten_embedding=True,depth = 2)
            self.merge_conv = Conv(2*512, 512, 1)
        self.conv4 = Conv(512,512, k=3, s=2, p=1)
        self.c3_3 = C3k2(c1=512,c2=512,c3k= True,n=2)  

        self.conv5 = Conv(512,512, 3, s=2, p=1)
        self.c3_4 = C3k2(512,512,c3k= True, n=2)
        # Transformer branch
        # if self.ace:
        #     self.ace = HyperACE(
        #                             c1=c(512),
        #                             c2=c(512),
        #                             n=2,
        #                        )
        #     self.fullpad = FullPAD_Tunnel()
        # else:
        self.sppf = SPPF( 512,512,5, 3, True)
        self.c2psa = C2PSA(512,512)
            

        # Neck: Feature aggregation and pyramid creation
        
        #self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        #self.reduce1 = Conv(c(1024) + c(512), c(512), 1)
        self.fuse1 = C3k2(c1=1024,c2=512,c3k= True, n=2) 

        #self.upsample2 = nn.Upsample(size=(75, 120), mode='nearest')
        #self.reduce2 = Conv(c(512) + c(256), c(256), k=1)
        self.fuse2 = C3k2(c1=1024,c2=256,c3k= True,n=2) 

        # Bottom-up pathway
        self.down1 = Conv(256, 256, k=3, s=2, p=1,)
        #self.merge1 = Conv(c(512) + c(512), c(512), k=1)
        self.fuse3 = C3k2(c1=768,c2=512,e=0.5,c3k= True,n=2)

        self.down2 = Conv(512, 512, k=3, s=2, p=1,)
        #self.merge2 = Conv(c(1024) + c(1024), c(1024), k=1)
        self.fuse4 = C3k2(c1=1024,c2=512,attn=True, n=1)

        # =========================
        # Temporal Fusion (SCA)
        # =========================
        self.join0 = Conv(256, 256//2, 1)
        self.join1 = Conv(512, 512//2, 1)
        self.join2 = Conv(512, 512//2, 1)

        self.improver0 = C3k2(c1=256,c2=256,c3k= True,n=2) 
        self.improver1  = C3k2(c1=512,c2=512,c3k= True,n=2)
        self.improver2  = C3k2(c1=512,c2=512,c3k= True, n=2)
        self._init_weights(self.join0)
        self._init_weights(self.join1)
        self._init_weights(self.join2)
        if self.improver:
            self._init_weights(self.improver0)
            self._init_weights(self.improver1)
            self._init_weights(self.improver2)
        if self.global_attn:
            self._init_weights(self.na)
            self._init_weights(self.merge_conv)


        # self.spatial_imp_1 = ImplicitShift_S(round(self.h/8), round(self.w/8), num_imp=self.num_imp_s, gram_mode=self.gram_mode,scale=self.scale)
        # self.spatial_imp_2 = ImplicitShift_S(round(self.h/16), round(self.w/16), num_imp=self.num_imp_s, gram_mode=self.gram_mode,scale=self.scale)
        # self.spatial_imp_3 = ImplicitShift_S(round(self.h/32), round(self.w/32), num_imp=self.num_imp_s, gram_mode=self.gram_mode,scale=self.scale)
            

        # =========================
        # Detect Head
        # =========================
        self.detect = Detect(nc=nc, reg_max = reg_max, ch=(256,512, 512))

    def _init_weights(self, module):
        """Initialize model weights to random values."""
        for m in module.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.1
                
            elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU,nn.GELU}:
                m.inplace = True

    # -------------------------------------------------
    # Single-frame backbone forward
    # -------------------------------------------------
    def off_forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        current = torch.split(input, 3, dim=1)[0]
        current_x1 = self.conv1(current)
        current_x2 = self.c3_1(self.conv2(current_x1))
        current_x3 = self.c3_2(self.conv3(current_x2))
        if self.global_attn:
            current_t = self.na(current)
            current_x3 = self.merge_conv(torch.cat([current_x3,current_t], 1))
        current_x4 = self.c3_3(self.conv4(current_x3))
        current_x5 = self.c3_4(self.conv5(current_x4))
        current_x5 = self.c2psa(self.sppf(current_x5))
        current_x6 = self.fuse1(torch.cat([F.interpolate(current_x5, size=current_x4.shape[2:4], mode='nearest'), current_x4], 1))
        current_p0 = self.fuse2(torch.cat([F.interpolate(current_x6, size=current_x3.shape[2:4], mode='nearest'), current_x3], 1))
        current_p1 = self.fuse3(torch.cat([self.down1(current_p0),current_x6], 1))
        current_p2 = self.fuse4(torch.cat([self.down2(current_p1), current_x5], 1))

        support = torch.split(input, 3, dim=1)[1]
        support_x1 = self.conv1(support)
        support_x2 = self.c3_1(self.conv2(support_x1))
        support_x3 = self.c3_2(self.conv3(support_x2))
        if self.global_attn:
            support_t = self.na(support)
            support_x3 = self.merge_conv(torch.cat([support_x3,support_t], 1))
        support_x4 = self.c3_3(self.conv4(support_x3))
        support_x5 = self.c3_4(self.conv5(support_x4))
        support_x5 = self.c2psa(self.sppf(support_x5))
        support_x6 = self.fuse1(torch.cat([F.interpolate(support_x5, size=support_x4.shape[2:4], mode='nearest'), support_x4], 1))
        support_p0 = self.fuse2(torch.cat([F.interpolate(support_x6, size=support_x3.shape[2:4], mode='nearest'), support_x3], 1))
        support_p1 = self.fuse3(torch.cat([self.down1(support_p0),support_x6], 1))
        support_p2 = self.fuse4(torch.cat([self.down2(support_p1), support_x5], 1))
        if self.improver:
            p0 = self.improver0(torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1)) + current_p0
            p1 = self.improver1(torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1)) + current_p1
            p2 = self.improver2(torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1)) + current_p2
        else:
            p0 = torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1) + current_p0
            p1 = torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1) + current_p1
            p2 = torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1) + current_p2
            
        outputs = [p0,p1,p2]

        return outputs

    def online_forward(self, input, buffer=None, node='star'):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        current_x1 = self.conv1(input)
        current_x2 = self.c3_1(self.conv2(current_x1))
        current_x3 = self.c3_2(self.conv3(current_x2))
        if self.global_attn:
            current_t = self.na(input)
            current_x3 = self.merge_conv(torch.cat([current_x3,current_t], 1))
        current_x4 = self.c3_3(self.conv4(current_x3))
        current_x5 = self.c3_4(self.conv5(current_x4))
        current_x5 = self.c2psa(self.sppf(current_x5))
        current_x6 = self.fuse1(torch.cat([F.interpolate(current_x5, size=current_x4.shape[2:4], mode='nearest'), current_x4], 1))
        current_p0 = self.fuse2(torch.cat([F.interpolate(current_x6, size=current_x3.shape[2:4], mode='nearest'), current_x3], 1))
        current_p1 = self.fuse3(torch.cat([self.down1(current_p0),current_x6], 1))
        current_p2 = self.fuse4(torch.cat([self.down2(current_p1), current_x5], 1))
        
        #####
        if node=='star':
            if self.improver:
                p0 = self.improver0(torch.cat([self.join0(current_p0), self.join0(current_p0)], dim=1)) + current_p0
                p1 = self.improver1(torch.cat([self.join1(current_p1), self.join1(current_p1)], dim=1)) + current_p1
                p2 = self.improver2(torch.cat([self.join2(current_p2), self.join2(current_p2)], dim=1)) + current_p2
            else:
                p0 = torch.cat([self.join0(current_p0), self.join0(current_p0)], dim=1) + current_p0
                p1 = torch.cat([self.join1(current_p1), self.join1(current_p1)], dim=1) + current_p1
                p2 = torch.cat([self.join2(current_p2), self.join2(current_p2)], dim=1) + current_p2

        elif node=='buffer':
            [support_p0, support_p1, support_p2] = buffer

            if self.improver:
                p0 = self.improver0(torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1)) + current_p0
                p1 = self.improver1(torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1)) + current_p1
                p2 = self.improver2(torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1)) + current_p2
            else:
                p0 = torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1) + current_p0
                p1 = torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1) + current_p1
                p2 = torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1) + current_p2

        outputs = [p0,p1,p2]
        buffer_ = [current_p0,current_p1,current_p2]

        return outputs, buffer_

    def forward(self, input, buffer=None, mode='off_pipe'):

        if mode=='off_pipe':
            # Glops caculate mode
            if input.size()[1] == 3:
                input = torch.cat([input, input], dim=1)
                output = self.off_forward(input)
            # offline train mode
            elif input.size()[1] == 6:
                output = self.off_forward(input)
            
            return self.detect(output)
        
        elif mode=='on_pipe':
            # online star state
            if buffer == None:
                output, buffer_ = self.online_forward(input, node='star')
            # online inference
            else:
                assert len(buffer) == 3
                assert input.size()[1] == 3
                output, buffer_ = self.online_forward(input, buffer=buffer, node='buffer')
            
            return self.detect(output), buffer_ 

class yolox(nn.Module):
    def __init__(self,
                 nc=8, 
                 reg_max=16,
                 global_attn = False,
                 improver = False):
                 
        super().__init__()

        self.global_attn = global_attn
        self.improver = improver


        # CNN stem
        self.conv1 = Conv(3,96, k=3, s=2, p=1)
        self.conv2 = Conv(96,192, k=3, s=2, p=1)
        self.c3_1 = C3k2(192,384,e=0.25, c3k=True, n=2)
        self.conv3 = Conv(384, 384, 3, s=2, p=1)
        self.c3_2 = C3k2(c1=384,c2=768,e=0.25,c3k= True, n=2) 

        
         # Transformer branch
        if self.global_attn:
            self.na = NAVisionTransformer(embed_dim=768,num_heads=4,patch_size=8,flatten_embedding=True,depth = 2)
            self.merge_conv = Conv(2*768, 768, 1)
        self.conv4 = Conv(768,768, k=3, s=2, p=1)
        self.c3_3 = C3k2(c1=768,c2=768,c3k= True,n=2)  

        self.conv5 = Conv(768,768, 3, s=2, p=1)
        self.c3_4 = C3k2(768,768,c3k= True, n=2)
        # Transformer branch
        # if self.ace:
        #     self.ace = HyperACE(
        #                             c1=c(512),
        #                             c2=c(512),
        #                             n=2,
        #                        )
        #     self.fullpad = FullPAD_Tunnel()
        # else:
        self.sppf = SPPF(768,768,5, 3, True)
        self.c2psa = C2PSA(768,768)
            

        # Neck: Feature aggregation and pyramid creation
        
        #self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        #self.reduce1 = Conv(c(1024) + c(512), c(512), 1)
        self.fuse1 = C3k2(c1=1536,c2=768,c3k= True, n=2) 

        #self.upsample2 = nn.Upsample(size=(75, 120), mode='nearest')
        #self.reduce2 = Conv(c(512) + c(256), c(256), k=1)
        self.fuse2 = C3k2(c1=1536,c2=384,c3k= True,n=2) 

        # Bottom-up pathway
        self.down1 = Conv(384, 384, k=3, s=2, p=1,)
        #self.merge1 = Conv(c(512) + c(512), c(512), k=1)
        self.fuse3 = C3k2(c1=1152,c2=768,e=0.5,c3k= True,n=2)

        self.down2 = Conv(768,768, k=3, s=2, p=1,)
        #self.merge2 = Conv(c(1024) + c(1024), c(1024), k=1)
        self.fuse4 = C3k2(c1=1536,c2=768,attn=True, n=1)

        # =========================
        # Temporal Fusion (SCA)
        # =========================
        self.join0 = Conv(384, 384//2, 1)
        self.join1 = Conv(768, 768//2, 1)
        self.join2 = Conv(768, 768//2, 1)

        self.improver0 = C3k2(c1=384,c2=384,c3k= True,n=2) 
        self.improver1 = C3k2(c1=768,c2=768,c3k= True,n=2)
        self.improver2 = C3k2(c1=768,c2=768,c3k= True, n=2)
        self._init_weights(self.join0)
        self._init_weights(self.join1)
        self._init_weights(self.join2)
        if self.improver:
            self._init_weights(self.improver0)
            self._init_weights(self.improver1)
            self._init_weights(self.improver2)
        if self.global_attn:
            self._init_weights(self.na)
            self._init_weights(self.merge_conv)


        # self.spatial_imp_1 = ImplicitShift_S(round(self.h/8), round(self.w/8), num_imp=self.num_imp_s, gram_mode=self.gram_mode,scale=self.scale)
        # self.spatial_imp_2 = ImplicitShift_S(round(self.h/16), round(self.w/16), num_imp=self.num_imp_s, gram_mode=self.gram_mode,scale=self.scale)
        # self.spatial_imp_3 = ImplicitShift_S(round(self.h/32), round(self.w/32), num_imp=self.num_imp_s, gram_mode=self.gram_mode,scale=self.scale)
            

        # =========================
        # Detect Head
        # =========================
        self.detect = Detect(nc=nc, reg_max = reg_max, ch=(384,768,768))

    def _init_weights(self, module):
        """Initialize model weights to random values."""
        for m in module.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.1
                
            elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU,nn.GELU}:
                m.inplace = True

    # -------------------------------------------------
    # Single-frame backbone forward
    # -------------------------------------------------
    def off_forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        current = torch.split(input, 3, dim=1)[0]
        current_x1 = self.conv1(current)
        current_x2 = self.c3_1(self.conv2(current_x1))
        current_x3 = self.c3_2(self.conv3(current_x2))
        if self.global_attn:
            current_t = self.na(current)
            current_x3 = self.merge_conv(torch.cat([current_x3,current_t], 1))
        current_x4 = self.c3_3(self.conv4(current_x3))
        current_x5 = self.c3_4(self.conv5(current_x4))
        current_x5 = self.c2psa(self.sppf(current_x5))
        current_x6 = self.fuse1(torch.cat([F.interpolate(current_x5, size=current_x4.shape[2:4], mode='nearest'), current_x4], 1))
        current_p0 = self.fuse2(torch.cat([F.interpolate(current_x6, size=current_x3.shape[2:4], mode='nearest'), current_x3], 1))
        current_p1 = self.fuse3(torch.cat([self.down1(current_p0),current_x6], 1))
        current_p2 = self.fuse4(torch.cat([self.down2(current_p1), current_x5], 1))

        support = torch.split(input, 3, dim=1)[1]
        support_x1 = self.conv1(support)
        support_x2 = self.c3_1(self.conv2(support_x1))
        support_x3 = self.c3_2(self.conv3(support_x2))
        if self.global_attn:
            support_t = self.na(support)
            support_x3 = self.merge_conv(torch.cat([support_x3,support_t], 1))
        support_x4 = self.c3_3(self.conv4(support_x3))
        support_x5 = self.c3_4(self.conv5(support_x4))
        support_x5 = self.c2psa(self.sppf(support_x5))
        support_x6 = self.fuse1(torch.cat([F.interpolate(support_x5, size=support_x4.shape[2:4], mode='nearest'), support_x4], 1))
        support_p0 = self.fuse2(torch.cat([F.interpolate(support_x6, size=support_x3.shape[2:4], mode='nearest'), support_x3], 1))
        support_p1 = self.fuse3(torch.cat([self.down1(support_p0),support_x6], 1))
        support_p2 = self.fuse4(torch.cat([self.down2(support_p1), support_x5], 1))
        if self.improver:
            p0 = self.improver0(torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1)) + current_p0
            p1 = self.improver1(torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1)) + current_p1
            p2 = self.improver2(torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1)) + current_p2
        else:
            p0 = torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1) + current_p0
            p1 = torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1) + current_p1
            p2 = torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1) + current_p2
            
        outputs = [p0,p1,p2]

        return outputs

    def online_forward(self, input, buffer=None, node='star'):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        current_x1 = self.conv1(input)
        current_x2 = self.c3_1(self.conv2(current_x1))
        current_x3 = self.c3_2(self.conv3(current_x2))
        if self.global_attn:
            current_t = self.na(input)
            current_x3 = self.merge_conv(torch.cat([current_x3,current_t], 1))
        current_x4 = self.c3_3(self.conv4(current_x3))
        current_x5 = self.c3_4(self.conv5(current_x4))
        current_x5 = self.c2psa(self.sppf(current_x5))
        current_x6 = self.fuse1(torch.cat([F.interpolate(current_x5, size=current_x4.shape[2:4], mode='nearest'), current_x4], 1))
        current_p0 = self.fuse2(torch.cat([F.interpolate(current_x6, size=current_x3.shape[2:4], mode='nearest'), current_x3], 1))
        current_p1 = self.fuse3(torch.cat([self.down1(current_p0),current_x6], 1))
        current_p2 = self.fuse4(torch.cat([self.down2(current_p1), current_x5], 1))
        
        #####
        if node=='star':
            if self.improver:
                p0 = self.improver0(torch.cat([self.join0(current_p0), self.join0(current_p0)], dim=1)) + current_p0
                p1 = self.improver1(torch.cat([self.join1(current_p1), self.join1(current_p1)], dim=1)) + current_p1
                p2 = self.improver2(torch.cat([self.join2(current_p2), self.join2(current_p2)], dim=1)) + current_p2
            else:
                p0 = torch.cat([self.join0(current_p0), self.join0(current_p0)], dim=1) + current_p0
                p1 = torch.cat([self.join1(current_p1), self.join1(current_p1)], dim=1) + current_p1
                p2 = torch.cat([self.join2(current_p2), self.join2(current_p2)], dim=1) + current_p2

        elif node=='buffer':
            [support_p0, support_p1, support_p2] = buffer

            if self.improver:
                p0 = self.improver0(torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1)) + current_p0
                p1 = self.improver1(torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1)) + current_p1
                p2 = self.improver2(torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1)) + current_p2
            else:
                p0 = torch.cat([self.join0(current_p0), self.join0(support_p0)], dim=1) + current_p0
                p1 = torch.cat([self.join1(current_p1), self.join1(support_p1)], dim=1) + current_p1
                p2 = torch.cat([self.join2(current_p2), self.join2(support_p2)], dim=1) + current_p2

        outputs = [p0,p1,p2]
        buffer_ = [current_p0,current_p1,current_p2]

        return outputs, buffer_

    def forward(self, input, buffer=None, mode='off_pipe'):

        if mode=='off_pipe':
            # Glops caculate mode
            if input.size()[1] == 3:
                input = torch.cat([input, input], dim=1)
                output = self.off_forward(input)
            # offline train mode
            elif input.size()[1] == 6:
                output = self.off_forward(input)
            
            return self.detect(output)
        
        elif mode=='on_pipe':
            # online star state
            if buffer == None:
                output, buffer_ = self.online_forward(input, node='star')
            # online inference
            else:
                assert len(buffer) == 3
                assert input.size()[1] == 3
                output, buffer_ = self.online_forward(input, buffer=buffer, node='buffer')
            
            return self.detect(output), buffer_ 
model = yolox(reg_max=1, nc=8, improver=True)
