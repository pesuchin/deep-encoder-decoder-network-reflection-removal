import chainer
import chainer.functions as F
import chainer.links as L


class ReflectionRemovalNet(chainer.Chain):
    def __init__(self, lambda_variable=0.001):
        super(ReflectionRemovalNet, self).__init__()
        self.lambda_variable = lambda_variable
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 64, stride=1, pad=1)
            self.conv2 = L.Convolution2D(None, 64, 64, stride=1, pad=1)
            self.conv3 = L.Convolution2D(None, 128, 64, stride=1, pad=1)
            self.conv4 = L.Convolution2D(None, 128, 64, stride=1, pad=1)
            self.conv5 = L.Convolution2D(None, 256, 64, stride=1, pad=1)
            self.conv6 = L.Convolution2D(None, 256, 64, stride=1, pad=1)

            self.conv7 = L.Convolution2D(None, 256, 64, stride=1, pad=1)
            self.conv8 = L.Convolution2D(None, 512, 64, stride=1, pad=1)
            self.conv9 = L.Convolution2D(None, 512, 64, stride=1, pad=1)
            self.conv10 = L.Convolution2D(None, 512, 64, stride=1, pad=1)
            self.conv11 = L.Convolution2D(None, 512, 64, stride=1, pad=1)
            self.conv12 = L.Convolution2D(None, 512, 64, stride=1, pad=1)

            self.deconv1 = L.Deconvolution2D(None, 512, 64, stride=2, pad=1)
            self.deconv2 = L.Deconvolution2D(None, 256, 64, stride=2, pad=1)
            self.deconv3 = L.Deconvolution2D(None, 128, 64, stride=2, pad=1)
            self.deconv4 = L.Deconvolution2D(None, 64, 64, stride=3, pad=1)
            self.deconv5 = L.Deconvolution2D(None, 1, 64, stride=2, pad=1)
            self.deconv6 = L.Deconvolution2D(None, 128, 64, stride=2, pad=1)
            self.deconv7 = L.Deconvolution2D(None, 64, 64, stride=2, pad=1)
            self.deconv8 = L.Deconvolution2D(None, 1, 64, stride=3, pad=1)
            self.deconv9 = L.Deconvolution2D(None, 256, 64, stride=2, pad=1)
            self.deconv10 = L.Deconvolution2D(None, 128, 64, stride=2, pad=1)
            self.deconv11 = L.Deconvolution2D(None, 64, 64, stride=2, pad=1)
            self.deconv12 = L.Deconvolution2D(None, 1, 64, stride=3, pad=1)

    def __call__(self, input_img, target_img):
        predicted_transmission_img = self.forward(input_img)

        l2_loss = F.mean_squared_error(predicted_transmission_img, target_img)

        # VGG loss: VGG19を特徴抽出器として用いて、予測された透過画像と目的の透過画像から特徴を抽出した特徴空間上のMSEをlossに採用
        vgg19 = L.VGG19Layers()
        extract_layers = ['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4', 'conv5_4']
        predicted_transmission_img_on_vgg19_results = vgg19.extract(predicted_transmission_img, layers=extract_layers)
        target_img_on_vgg19_results = vgg19.extract(target_img, layers=extract_layers)

        vgg_loss = 0
        for extract_layer in extract_layers:
            phi_F = predicted_transmission_img_on_vgg19_results[extract_layer]
            phi_aT = target_img_on_vgg19_results[extract_layer]

            vgg_loss += F.mean_squared_error(phi_F, phi_aT) / (phi_F.shape[2] * phi_F.shape[3])

        loss = l2_loss + (self.lambda_variable * vgg_loss)

        return loss

    def feature_extraction(self, img):
        feature = self.conv1(img)
        feature = self.conv2(feature)
        feature = self.conv3(feature)
        feature = self.conv4(feature)
        feature = self.conv5(feature)
        feature = self.conv6(feature)
        return feature

    def reflection_recovery_and_removal(self, feature):
        removal_feature1 = F.relu(feature)
        removal_feature1 = self.conv7(removal_feature1)
        removal_feature1 = self.conv8(removal_feature1)

        removal_feature2 = F.relu(removal_feature1)
        removal_feature2 = self.conv9(removal_feature2)
        removal_feature2 = self.conv10(removal_feature2)

        removal_feature3 = F.relu(removal_feature2)
        removal_feature3 = self.conv11(removal_feature3)
        removal_feature3 = self.conv12(removal_feature3)
        removal_feature3 = self.deconv1(removal_feature3)
        removal_feature3 = self.deconv2(removal_feature3)

        removal_2_plus_3 = removal_feature2 + removal_feature3
        removal_2_plus_3 = F.relu(removal_2_plus_3)
        removal_2_plus_3 = self.deconv3(removal_2_plus_3)
        removal_2_plus_3 = self.deconv4(removal_2_plus_3)

        removal_1_plus_2_plus_3 = removal_feature1 + removal_2_plus_3
        removal_1_plus_2_plus_3 = F.relu(removal_1_plus_2_plus_3)
        removal_1_plus_2_plus_3 = self.deconv5(removal_1_plus_2_plus_3)
        removal_1_plus_2_plus_3 = self.deconv6(removal_1_plus_2_plus_3)
        return removal_1_plus_2_plus_3

    def transmission_layer(self, feature, removal_1_plus_2_plus_3):
        diff_reflection = feature - removal_1_plus_2_plus_3
        diff_reflection = F.relu(diff_reflection)
        diff_reflection = self.deconv7(diff_reflection)
        diff_reflection = self.deconv8(diff_reflection)
        diff_reflection = self.deconv9(diff_reflection)
        diff_reflection = self.deconv10(diff_reflection)
        diff_reflection = self.deconv11(diff_reflection)
        diff_reflection = self.deconv12(diff_reflection)
        return diff_reflection

    def forward(self, img):
        feature = self.feature_extraction(img)
        removal_1_plus_2_plus_3 = self.reflection_recovery_and_removal(feature)
        transmission_img = self.transmission_layer(feature, removal_1_plus_2_plus_3)

        return transmission_img
