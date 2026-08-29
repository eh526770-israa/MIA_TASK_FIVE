import torch
import matplotlib.pyplot as plt


class Phase4:

    def __init__(
        self,
        model,
        test_loader,
        criterion,
        phase2,
        phase3,
        class_names,
        device
    ):

        # =====================================================
        # Existing objects from previous phases
        # =====================================================

        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.class_names = class_names
        self.device = device

        self.phase2 = phase2
        self.phase3 = phase3

        self.model.eval()

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
    # Find interesting examples
    #
    # Interesting example:
    # 1. Model is correct on clean image
    # 2. FGSM changes the prediction
    # =========================================================

    def find_interesting_examples(
        self,
        epsilon,
        number_of_examples=3,
        max_search=200
    ):

        examples = []

        self.model.eval()

        checked = 0

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

                if checked >= max_search:

                    return examples

                checked += 1

                # We only attack correctly classified images
                if predictions[i].item() != labels[i].item():

                    continue

                image = images[i:i + 1].clone()
                label = labels[i:i + 1].clone()

                # Generate adversarial image
                adversarial_image = self.phase2.fgsm_attack(
                    image,
                    label,
                    epsilon
                )

                # Get adversarial prediction
                adversarial_prediction = (
                    self.get_prediction(
                        adversarial_image
                    )
                )

                # Keep only successful attacks
                if (
                    adversarial_prediction
                    != label.item()
                ):

                    examples.append(
                        {
                            "image": image,
                            "label": label,
                            "original_prediction": predictions[i].item(),
                            "adversarial_image": adversarial_image,
                            "adversarial_prediction": adversarial_prediction
                        }
                    )

                if len(examples) >= number_of_examples:

                    return examples

        return examples

    # =========================================================
    # Generate XAI for clean image
    # =========================================================

    def analyze_clean_image(self, image):

        # -----------------------------------------------------
        # Saliency Map
        # -----------------------------------------------------

        saliency, saliency_prediction = (
            self.phase3.saliency_map(
                image
            )
        )

        # -----------------------------------------------------
        # Grad-CAM
        # -----------------------------------------------------

        cam, cam_prediction = (
            self.phase3.grad_cam(
                image
            )
        )

        return (
            saliency,
            saliency_prediction,
            cam,
            cam_prediction
        )

    # =========================================================
    # Generate XAI for adversarial image
    # =========================================================

    def analyze_adversarial_image(
        self,
        adversarial_image
    ):

        # -----------------------------------------------------
        # Saliency Map
        # -----------------------------------------------------

        adversarial_saliency, saliency_prediction = (
            self.phase3.saliency_map(
                adversarial_image
            )
        )

        # -----------------------------------------------------
        # Grad-CAM
        # -----------------------------------------------------

        adversarial_cam, cam_prediction = (
            self.phase3.grad_cam(
                adversarial_image
            )
        )

        return (
            adversarial_saliency,
            saliency_prediction,
            adversarial_cam,
            cam_prediction
        )

    # =========================================================
    # Visualize forensic analysis
    # =========================================================

    def visualize_forensic_analysis(
        self,
        example,
        epsilon
    ):

        image = example["image"]

        label = example["label"]

        original_prediction = (
            example["original_prediction"]
        )

        adversarial_image = (
            example["adversarial_image"]
        )

        adversarial_prediction = (
            example["adversarial_prediction"]
        )

        # =====================================================
        # Clean XAI
        # =====================================================

        (
            clean_saliency,
            clean_saliency_prediction,
            clean_cam,
            clean_cam_prediction
        ) = self.analyze_clean_image(
            image
        )

        # =====================================================
        # Adversarial XAI
        # =====================================================

        (
            adversarial_saliency,
            adversarial_saliency_prediction,
            adversarial_cam,
            adversarial_cam_prediction
        ) = self.analyze_adversarial_image(
            adversarial_image
        )

        # =====================================================
        # Convert images to NumPy
        # =====================================================

        original = (
            image
            .squeeze(0)
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        adversarial = (
            adversarial_image
            .squeeze(0)
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        # =====================================================
        # Perturbation
        # =====================================================

        perturbation = (
            adversarial - original
        )

        perturbation_display = (
            perturbation
            - perturbation.min()
        )

        max_value = perturbation_display.max()

        if max_value > 0:

            perturbation_display = (
                perturbation_display
                / max_value
            )

        # =====================================================
        # Convert XAI maps
        # =====================================================

        clean_saliency = (
            clean_saliency
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        clean_cam = (
            clean_cam
            .detach()
            .cpu()
            .numpy()
        )

        adversarial_saliency = (
            adversarial_saliency
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        adversarial_cam = (
            adversarial_cam
            .detach()
            .cpu()
            .numpy()
        )

        # =====================================================
        # Visualization
        # =====================================================

        plt.figure(
            figsize=(18, 10)
        )

        # -----------------------------------------------------
        # 1. Clean image
        # -----------------------------------------------------

        plt.subplot(2, 4, 1)

        plt.imshow(original)

        plt.title(
            "Clean Image\n"
            f"True: "
            f"{self.class_names[label.item()]}\n"
            f"Prediction: "
            f"{self.class_names[original_prediction]}"
        )

        plt.axis("off")

        # -----------------------------------------------------
        # 2. Clean Saliency
        # -----------------------------------------------------

        plt.subplot(2, 4, 2)

        plt.imshow(
            clean_saliency,
            cmap="hot"
        )

        plt.title(
            "Clean - Saliency Map"
        )

        plt.axis("off")

        # -----------------------------------------------------
        # 3. Clean Grad-CAM
        # -----------------------------------------------------

        plt.subplot(2, 4, 3)

        plt.imshow(
            clean_cam,
            cmap="jet"
        )

        plt.title(
            "Clean - Grad-CAM"
        )

        plt.axis("off")

        # -----------------------------------------------------
        # 4. Clean Grad-CAM Overlay
        # -----------------------------------------------------

        plt.subplot(2, 4, 4)

        plt.imshow(original)

        plt.imshow(
            clean_cam,
            cmap="jet",
            alpha=0.5
        )

        plt.title(
            "Clean - Grad-CAM Overlay"
        )

        plt.axis("off")

        # -----------------------------------------------------
        # 5. Adversarial image
        # -----------------------------------------------------

        plt.subplot(2, 4, 5)

        plt.imshow(adversarial)

        plt.title(
            "Adversarial Image\n"
            f"Prediction: "
            f"{self.class_names[adversarial_prediction]}\n"
            f"Epsilon: {epsilon}"
        )

        plt.axis("off")

        # -----------------------------------------------------
        # 6. Perturbation
        # -----------------------------------------------------

        plt.subplot(2, 4, 6)

        plt.imshow(
            perturbation_display
        )

        plt.title(
            "Adversarial Perturbation"
        )

        plt.axis("off")

        # -----------------------------------------------------
        # 7. Adversarial Saliency
        # -----------------------------------------------------

        plt.subplot(2, 4, 7)

        plt.imshow(
            adversarial_saliency,
            cmap="hot"
        )

        plt.title(
            "Adversarial - Saliency"
        )

        plt.axis("off")

        # -----------------------------------------------------
        # 8. Adversarial Grad-CAM
        # -----------------------------------------------------

        plt.subplot(2, 4, 8)

        plt.imshow(adversarial)

        plt.imshow(
            adversarial_cam,
            cmap="jet",
            alpha=0.5
        )

        plt.title(
            "Adversarial - Grad-CAM"
        )

        plt.axis("off")

        plt.tight_layout()

        plt.show()

        # =====================================================
        # Print forensic information
        # =====================================================

        print()
        print("=" * 70)
        print("FORENSIC ANALYSIS")
        print("=" * 70)

        print(
            "True class:",
            self.class_names[label.item()]
        )

        print(
            "Original prediction:",
            self.class_names[
                original_prediction
            ]
        )

        print(
            "Adversarial prediction:",
            self.class_names[
                adversarial_prediction
            ]
        )

        print(
            "Epsilon:",
            epsilon
        )

        print(
            "Attack successful:",
            original_prediction
            != adversarial_prediction
        )

        print(
            "Clean Saliency prediction:",
            self.class_names[
                clean_saliency_prediction
            ]
        )

        print(
            "Clean Grad-CAM prediction:",
            self.class_names[
                clean_cam_prediction
            ]
        )

        print(
            "Adversarial Saliency prediction:",
            self.class_names[
                adversarial_saliency_prediction
            ]
        )

        print(
            "Adversarial Grad-CAM prediction:",
            self.class_names[
                adversarial_cam_prediction
            ]
        )

    # =========================================================
    # Run complete forensic analysis
    # =========================================================

    def run(
        self,
        epsilon,
        number_of_examples=3
    ):

        print()
        print("=" * 70)
        print("PHASE 4 - FORENSIC ANALYSIS")
        print("=" * 70)

        print(
            f"Searching for {number_of_examples} "
            f"successful adversarial examples..."
        )

        examples = self.find_interesting_examples(
            epsilon=epsilon,
            number_of_examples=number_of_examples
        )

        if len(examples) == 0:

            print()
            print(
                "No successful adversarial examples "
                "were found."
            )

            print(
                "Try using a larger epsilon."
            )

            return

        print()
        print(
            f"Found {len(examples)} "
            "successful examples."
        )

        # Analyze every example
        for index, example in enumerate(
            examples,
            start=1
        ):

            print()
            print("#" * 70)
            print(
                f"FORENSIC EXAMPLE {index}"
            )
            print("#" * 70)

            self.visualize_forensic_analysis(
                example,
                epsilon
            )

        return examples

