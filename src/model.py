import chainer
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import numpy as np


class ReflectionRemovalNet(chainer.Chain):
    def __init__(self, lambda_variable=0.001):
        super(ReflectionRemovalNet, self).__init__()
        self.lambda_variable = lambda_variable
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.conv2 = L.Convolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.conv3 = L.Convolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.conv4 = L.Convolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.conv5 = L.Convolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.conv6 = L.Convolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)

            self.bn1 = L.BatchNormalization(64)
            self.bn2 = L.BatchNormalization(64)
            self.bn3 = L.BatchNormalization(64)
            self.bn4 = L.BatchNormalization(64)
            self.bn5 = L.BatchNormalization(64)
            self.bn6 = L.BatchNormalization(64)
            self.bn7 = L.BatchNormalization(64)
            self.bn8 = L.BatchNormalization(64)
            self.bn9 = L.BatchNormalization(64)
            self.bn10 = L.BatchNormalization(64)
            self.bn11 = L.BatchNormalization(64)
            self.bn12 = L.BatchNormalization(64)
            self.bn13 = L.BatchNormalization(64)
            self.bn14 = L.BatchNormalization(64)
            self.bn15 = L.BatchNormalization(64)
            self.bn16 = L.BatchNormalization(64)
            self.bn17 = L.BatchNormalization(64)
            self.bn18 = L.BatchNormalization(64)
            self.bn19 = L.BatchNormalization(64)
            self.bn20 = L.BatchNormalization(64)
            self.bn21 = L.BatchNormalization(64)
            self.bn22 = L.BatchNormalization(64)
            self.bn23 = L.BatchNormalization(64)
            self.bn24 = L.BatchNormalization(3)

            self.conv7 = L.Convolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.conv8 = L.Convolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.conv8_skip = L.Convolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)

            self.conv9 = L.Convolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.conv10 = L.Convolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.conv10_skip = L.Convolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.conv11 = L.Convolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.conv12 = L.Convolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)

            self.deconv1 = L.Deconvolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.deconv2 = L.Deconvolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.deconv3 = L.Deconvolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.deconv4 = L.Deconvolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.deconv5 = L.Deconvolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.deconv6 = L.Deconvolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.deconv7 = L.Deconvolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.deconv8 = L.Deconvolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.deconv9 = L.Deconvolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.deconv10 = L.Deconvolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.deconv11 = L.Deconvolution2D(None, 64, ksize=(5, 5), stride=1, pad=1)
            self.deconv12 = L.Deconvolution2D(None, 3, ksize=(5, 5), stride=1, pad=1)

    def __call__(self, input_img, target_img):
        predicted_transmission_img = self.forward(input_img)

        l2_loss = F.mean_squared_error(predicted_transmission_img, target_img)

        loss = l2_loss
        chainer.report({'loss': loss}, self)

        return loss

    def feature_extraction(self, img):
        feature = self.conv1(img)
        feature = self.bn1(feature)
        feature = F.relu(feature)
        feature = self.conv2(feature)
        feature = self.bn2(feature)
        feature = F.relu(feature)
        feature = self.conv3(feature)
        feature = self.bn3(feature)
        feature = F.relu(feature)
        feature = self.conv4(feature)
        feature = self.bn4(feature)
        feature = F.relu(feature)
        feature = self.conv5(feature)
        feature = self.bn5(feature)
        feature = F.relu(feature)
        feature = self.conv6(feature)
        feature = self.bn6(feature)
        return feature

    def reflection_recovery_and_removal(self, feature):
        removal_feature1 = F.relu(feature)
        removal_feature1 = self.conv7(removal_feature1)
        removal_feature1 = self.bn7(removal_feature1)
        removal_feature1 = F.relu(removal_feature1)
        removal_feature1 = self.conv8(removal_feature1)
        removal_feature1 = self.bn8(removal_feature1)

        removal_feature2 = F.relu(removal_feature1)
        removal_feature2 = self.conv9(removal_feature2)
        removal_feature2 = self.bn9(removal_feature2)
        removal_feature2 = F.relu(removal_feature2)
        removal_feature2 = self.conv10(removal_feature2)
        removal_feature2 = self.bn10(removal_feature2)

        removal_feature3 = F.relu(removal_feature2)
        removal_feature3 = self.conv11(removal_feature3)
        removal_feature3 = self.bn11(removal_feature3)
        removal_feature3 = F.relu(removal_feature3)
        removal_feature3 = self.conv12(removal_feature3)
        removal_feature3 = self.bn12(removal_feature3)
        removal_feature3 = F.relu(removal_feature3)
        removal_feature3 = self.deconv1(removal_feature3)
        removal_feature3 = self.bn13(removal_feature3)
        removal_feature3 = F.relu(removal_feature3)
        removal_feature3 = self.deconv2(removal_feature3)
        removal_feature3 = self.bn14(removal_feature3)
        removal_feature3 = F.relu(removal_feature3)

        removal_2_plus_3 = removal_feature2 + removal_feature3
        removal_2_plus_3 = F.relu(removal_2_plus_3)
        removal_2_plus_3 = self.deconv3(removal_2_plus_3)
        removal_2_plus_3 = self.bn15(removal_2_plus_3)
        removal_2_plus_3 = F.relu(removal_2_plus_3)
        removal_2_plus_3 = self.deconv4(removal_2_plus_3)
        removal_2_plus_3 = self.bn16(removal_2_plus_3)
        removal_2_plus_3 = F.relu(removal_2_plus_3)

        removal_1_plus_2_plus_3 = removal_feature1 + removal_2_plus_3
        removal_1_plus_2_plus_3 = F.relu(removal_1_plus_2_plus_3)
        removal_1_plus_2_plus_3 = self.deconv5(removal_1_plus_2_plus_3)
        removal_1_plus_2_plus_3 = self.bn17(removal_1_plus_2_plus_3)
        removal_1_plus_2_plus_3 = F.relu(removal_1_plus_2_plus_3)
        removal_1_plus_2_plus_3 = self.deconv6(removal_1_plus_2_plus_3)
        removal_1_plus_2_plus_3 = self.bn18(removal_1_plus_2_plus_3)
        removal_1_plus_2_plus_3 = F.relu(removal_1_plus_2_plus_3)
        return removal_1_plus_2_plus_3

    def transmission_layer(self, feature, removal_1_plus_2_plus_3):
        diff_reflection = F.relu(feature - removal_1_plus_2_plus_3)
        diff_reflection = self.deconv7(diff_reflection)
        diff_reflection = self.bn19(diff_reflection)
        diff_reflection = F.relu(diff_reflection)
        diff_reflection = self.deconv8(diff_reflection)
        diff_reflection = self.bn20(diff_reflection)
        diff_reflection = F.relu(diff_reflection)
        diff_reflection = self.deconv9(diff_reflection)
        diff_reflection = self.bn21(diff_reflection)
        diff_reflection = F.relu(diff_reflection)
        diff_reflection = self.deconv10(diff_reflection)
        diff_reflection = self.bn22(diff_reflection)
        diff_reflection = F.relu(diff_reflection)
        diff_reflection = self.deconv11(diff_reflection)
        diff_reflection = self.bn23(diff_reflection)
        diff_reflection = F.relu(diff_reflection)
        diff_reflection = self.deconv12(diff_reflection)
        diff_reflection = self.bn24(diff_reflection)
        diff_reflection = F.relu(diff_reflection)
        return diff_reflection

    def forward(self, img):
        feature = self.feature_extraction(img)
        removal_1_plus_2_plus_3 = self.reflection_recovery_and_removal(feature)
        transmission_img = self.transmission_layer(feature, removal_1_plus_2_plus_3)

        return transmission_img
