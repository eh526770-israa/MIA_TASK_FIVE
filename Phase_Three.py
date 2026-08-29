import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Phase3:

    def __init__(self, model, test_loader, class_names, device):

        # =====================================================
        # Use the trained model and test data from Phase 1
        # =====================================================

        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device

        # XAI does not train the model
        self.model.eval()

        # Variables used by Grad-CAM
        self.activations = None
        self.gradients = None

    # =========================================================
    # Get a correctly classified image
    # =========================================================

    def get_correct_image(self):

        self.model.eval()

        for images, labels in self.test_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():

                outputs = self.model(images)

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

            for i in range(len(images)):

                if predictions[i].item() == labels[i].item():

                    image = images[i:i + 1].clone()
                    label = labels[i:i + 1].clone()

                    print()
                    print("=" * 60)
                    print("IMAGE SELECTED")
                    print("=" * 60)

                    print(
                        "True class:",
                        self.class_names[label.item()]
                    )

                    print(
                        "Predicted class:",
                        self.class_names[
                            predictions[i].item()
                        ]
                    )

                    return image, label

        raise RuntimeError(
            "No correctly classified image was found."
        )

    # =========================================================
    # Get prediction
    # =========================================================

    def get_prediction(self, image):

        self.model.eval()

        with torch.no_grad():

            output = self.model(image)

            prediction = torch.argmax(
                output,
                dim=1
            ).item()

        return prediction

    # =========================================================
    # SALIENCY MAP
    # Vanilla Gradients
    # =========================================================

    def saliency_map(self, image):

        self.model.eval()

        # Create independent copy
        image = image.clone().detach()

        # Enable gradients for the input image
        image.requires_grad_(True)

        # Forward pass
        output = self.model(image)

        # Get predicted class
        predicted_class = torch.argmax(
            output,
            dim=1
        ).item()

        # Score corresponding to predicted class
        target_score = output[
            0,
            predicted_class
        ]

        # Clear previous gradients
        self.model.zero_grad()

        # Backpropagation
        target_score.backward()

        # Gradient with respect to image
        image_gradient = image.grad.detach()

        # Absolute value
        image_gradient = image_gradient.abs()

        # Convert 3 channels into one map
        saliency = image_gradient.max(
            dim=1
        )[0]

        # Normalize to [0, 1]
        saliency_min = saliency.min()
        saliency_max = saliency.max()

        saliency = (
            saliency - saliency_min
        ) / (
            saliency_max - saliency_min + 1e-8
        )

        return saliency, predicted_class

    # =========================================================
    # Grad-CAM Hooks
    # =========================================================

    def save_activation(self, module, input, output):

        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0].detach()

    # =========================================================
    # Register hooks on ResNet-34 layer4
    # =========================================================

    def register_gradcam_hooks(self):

        # ResNet-34:
        # layer4 is the last convolutional block

        target_layer = self.model.layer4

        target_layer.register_forward_hook(
            self.save_activation
        )

        target_layer.register_full_backward_hook(
            self.save_gradient
        )

    # =========================================================
    # GRAD-CAM
    # =========================================================

    def grad_cam(self, image):

        self.model.eval()

        # Make sure hooks exist
        self.register_gradcam_hooks()

        # Clear old stored values
        self.activations = None
        self.gradients = None

        # Forward pass
        output = self.model(image)

        # Get predicted class
        predicted_class = torch.argmax(
            output,
            dim=1
        ).item()

        # Score of predicted class
        target_score = output[
            0,
            predicted_class
        ]

        # Clear model gradients
        self.model.zero_grad()

        # Backward pass
        target_score.backward()

        # Check that hooks captured the required values
        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:

            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # =====================================================
        # Calculate Grad-CAM
        # =====================================================

        # Activations shape:
        # [batch, channels, height, width]

        activations = self.activations

        gradients = self.gradients

        # Global average pooling of gradients
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted sum of feature maps
        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        # Keep positive influence only
        cam = F.relu(cam)

        # Resize CAM to image size
        cam = F.interpolate(
            cam,
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        # Remove channel dimension
        cam = cam.squeeze(
            0,
            1
        )

        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        return cam, predicted_class

    # =========================================================
    # Convert tensor image to NumPy image
    # =========================================================

    def tensor_to_image(self, image):

        image = image.squeeze(
            0
        ).detach().cpu()

        image = image.permute(
            1,
            2,
            0
        ).numpy()

        return image

    # =========================================================
    # Visualize Saliency Map
    # =========================================================

    def visualize_saliency(
        self,
        image,
        saliency,
        predicted_class
    ):

        original = self.tensor_to_image(
            image
        )

        saliency = saliency.squeeze(
            0
        ).detach().cpu().numpy()

        plt.figure(
            figsize=(15, 5)
        )

        # Original image
        plt.subplot(
            1,
            3,
            1
        )

        plt.imshow(original)

        plt.title(
            "Original Image\n"
            f"Prediction: "
            f"{self.class_names[predicted_class]}"
        )

        plt.axis("off")

        # Saliency map
        plt.subplot(
            1,
            3,
            2
        )

        plt.imshow(
            saliency,
            cmap="hot"
        )

        plt.title(
            "Saliency Map"
        )

        plt.axis("off")

        # Overlay
        plt.subplot(
            1,
            3,
            3
        )

        plt.imshow(original)

        plt.imshow(
            saliency,
            cmap="hot",
            alpha=0.5
        )

        plt.title(
            "Saliency Overlay"
        )

        plt.axis("off")

        plt.tight_layout()

        plt.show()

    # =========================================================
    # Visualize Grad-CAM
    # =========================================================

    def visualize_gradcam(
        self,
        image,
        cam,
        predicted_class
    ):

        original = self.tensor_to_image(
            image
        )

        cam = cam.detach().cpu().numpy()

        plt.figure(
            figsize=(15, 5)
        )

        # Original image
        plt.subplot(
            1,
            3,
            1
        )

        plt.imshow(original)

        plt.title(
            "Original Image\n"
            f"Prediction: "
            f"{self.class_names[predicted_class]}"
        )

        plt.axis("off")

        # Grad-CAM heatmap
        plt.subplot(
            1,
            3,
            2
        )

        plt.imshow(
            cam,
            cmap="jet"
        )

        plt.title(
            "Grad-CAM"
        )

        plt.axis("off")

        # Overlay
        plt.subplot(
            1,
            3,
            3
        )

        plt.imshow(original)

        plt.imshow(
            cam,
            cmap="jet",
            alpha=0.5
        )

        plt.title(
            "Grad-CAM Overlay"
        )

        plt.axis("off")

        plt.tight_layout()

        plt.show()

    # =========================================================
    # Run Saliency Map
    # =========================================================

    def run_saliency(self):

        print()
        print("=" * 60)
        print("SALiency MAP")
        print("=" * 60)

        # Get correctly classified image
        image, label = self.get_correct_image()

        # Generate saliency
        saliency, predicted_class = self.saliency_map(
            image
        )

        # Visualize
        self.visualize_saliency(
            image,
            saliency,
            predicted_class
        )

        return (
            image,
            label,
            saliency,
            predicted_class
        )

    # =========================================================
    # Run Grad-CAM
    # =========================================================

    def run_gradcam(self, image=None):

        print()
        print("=" * 60)
        print("GRAD-CAM")
        print("=" * 60)

        # If no image was provided,
        # get a correctly classified image
        if image is None:

            image, label = self.get_correct_image()

        # Generate Grad-CAM
        cam, predicted_class = self.grad_cam(
            image
        )

        # Visualize
        self.visualize_gradcam(
            image,
            cam,
            predicted_class
        )

        return (
            cam,
            predicted_class
        )

    # =============================================================
    # Run Phase 3
    # =============================================================

    def run(self):

        print("\n" + "=" * 60)
        print("PHASE 3 - MODEL EXPLAINABILITY")
        print("=" * 60)

        image, label = self.get_correct_image()

        saliency, saliency_prediction = self.saliency_map(image)

        self.visualize_saliency(
            image,
            saliency,
            saliency_prediction
        )

        cam, cam_prediction = self.grad_cam(image)

        self.visualize_gradcam(
            image,
            cam,
            cam_prediction
        )

        print("\nPhase 3 completed successfully!")

        return {
            "model": self.model,
            "device": self.device,
            "image": image,
            "label": label,
            "saliency": saliency,
            "saliency_prediction": saliency_prediction,
            "gradcam": cam,
            "gradcam_prediction": cam_prediction
        }
