# AI PC模型用途与公开分发询问草稿（未发送）

这些是用户审核用文本，不代表已联系上游或取得授权。实际比赛主办方、商业/非商业用途、公开位置由发送者据实补充；不要未经确认承诺全部用途属于非商业。两份草稿不附原始照片或权重。

## Office-Home／相关照片权利人

Subject: Clarification of research use and redistribution of fine-tuned model weights using selected Office-Home images

Hello,

We are developing an academic robotics competition project on an AMD AI PC. Our application detects six tabletop object categories and estimates visible grasp surfaces. We have selected 32 Office-Home eraser images in our current data manifests: 18 training and 14 development-validation images. Four original training images have visible-surface polygon labels; train-only crops and synthetic derivatives are also used. Our current models are a fine-tuned Ultralytics YOLOv8s detector and a MobileNetV3-based surface network. Other project and public data are also present.

We have read the Fair Use Notice at https://www.hemanthdv.org/officeHomeDataset.html. We understand that it is not a blanket redistribution license, and that permission may need to come directly from the underlying copyright owners.

Before releasing anything, could you clarify:

1. Whether the relevant rights cover use of these selected images and train-only crops/composites for the described competition research and public demonstration.
2. Whether publicly distributing our resulting fine-tuned numerical model weights, application source, model evaluation numbers and appropriately selected demo outputs is covered. We plan to license team-original application code under AGPL-3.0; we do not seek to relicense the underlying photographs or imply that AGPL changes their terms.
3. Whether the original images and their derived training rasters must remain undistributed, and what attribution or other conditions should accompany any permitted model or demo release.
4. If you cannot grant or confirm the necessary rights for the selected photographs, whether you can identify their copyright holders or an authoritative route for seeking permission.

We can provide a private list of selected source paths and SHA256 identifiers if appropriate. We will not describe your reply as covering other people's rights or uses beyond what you explicitly authorize.

Thank you.

## TorchVision MobileNetV3-Small预训练参数

Subject: Licensing clarification for MobileNet_V3_Small_Weights.IMAGENET1K_V1 and a fine-tuned robotics surface model

Hello,

Our academic robotics competition project uses a custom visible-surface decoder initialized with the TorchVision MobileNetV3-Small encoder from MobileNet_V3_Small_Weights.IMAGENET1K_V1. The official enumeration points to https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth. We are not using SWAG weights.

We understand that TorchVision software's BSD license does not by itself resolve all pretrained-weight or training-dataset terms. We also understand the model documentation's notice that pretrained models may have their own terms derived from the training data.

Could you point us to the authoritative terms applying to this exact pretrained weight release, including whether they cover academic competition use and public redistribution of a fine-tuned state_dict containing the encoder? What attribution and other conditions apply? If you cannot grant rights concerning the original training data, please distinguish the rights you can confirm from those that require a separate rights holder's permission.

Our team-original application code will use AGPL-3.0. We will not redistribute ImageNet training images, and we do not claim that our code license changes any upstream rights. We will describe the actual intended demo/distribution scope accurately before relying on a permission statement.

Thank you.

## 使用说明

以上问询需用户明确决定是否发送及通过哪个已核实的官方渠道。本轮没有发送邮件、GitHub issue或其他对外消息。Office-Home组织者的答复不自动代表原照片版权人同意；TorchVision软件维护者的答复也不自动授予其并不拥有的数据权利。
