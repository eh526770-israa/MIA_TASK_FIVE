import torch
import matplotlib.pyplot as plt


class Phase2:



    def __init__(
        self,
        model,
        test_loader,
        criterion,
        device
    ):

        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
 
 

        # Loss function used by FGSM
        self.criterion = torch.nn.CrossEntropyLoss()

        # We are attacking the trained model,
        # not training it again
        self.model.eval()

    # =========================================================
    # 1. Get model prediction
    # =========================================================

    def get_prediction(self, image):

        self.model.eval()

        with torch.no_grad():

            output = self.model(image)

            prediction = torch.argmax(
                output,
                dim=1
            )

        return prediction

    # =========================================================
    # 2. Find correctly classified test images
    # =========================================================

    def get_correct_images(self, max_images=100):

        correct_images = []

        self.model.eval()

        for images, labels in self.test_loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            # Make predictions
            with torch.no_grad():

                outputs = self.model(images)

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

            # Check every image in the batch
            for i in range(len(images)):

                if predictions[i].item() == labels[i].item():

                    correct_images.append(
                        (
                            images[i:i + 1].clone(),
                            labels[i:i + 1].clone()
                        )
                    )

                    # Stop when we have enough images
                    if len(correct_images) >= max_images:

                        return correct_images

        return correct_images

    # =========================================================
    # 3. FGSM Attack
    # =========================================================

    def fgsm_attack(
        self,
        image,
        label,
        epsilon
    ):

        # Create a separate copy of the image
        image = image.clone().detach()

        # We need the gradient with respect to the image
        image.requires_grad = True

        # Forward pass
        output = self.model(image)

        # Calculate loss
        loss = self.criterion(
            output,
            label
        )

        # Remove old gradients from the model
        self.model.zero_grad()

        # Backpropagation
        loss.backward()

        # Get gradient of the image
        image_gradient = image.grad.data

        # Get only the direction of the gradient
        gradient_sign = image_gradient.sign()

        # Generate adversarial image
        adversarial_image = (
            image
            + epsilon * gradient_sign
        )

        # Keep pixel values in valid range
        adversarial_image = torch.clamp(
            adversarial_image,
            0,
            1
        )

        # We don't need gradients anymore
        adversarial_image = adversarial_image.detach()

        return adversarial_image

    # =========================================================
    # 4. Attack one image
    # =========================================================

    def attack_one_image(
        self,
        image,
        label,
        epsilon
    ):

        # Original prediction
        original_prediction = self.get_prediction(
            image
        )

        # Generate adversarial example
        adversarial_image = self.fgsm_attack(
            image,
            label,
            epsilon
        )

        # Prediction after attack
        adversarial_prediction = self.get_prediction(
            adversarial_image
        )

        # Check if attack succeeded
        attack_successful = (
            adversarial_prediction.item()
            != label.item()
        )

        return (
            adversarial_image,
            original_prediction,
            adversarial_prediction,
            attack_successful
        )

    # =========================================================
    # 5. Test one epsilon
    # =========================================================

    def evaluate_epsilon(
        self,
        epsilon,
        max_images=100
    ):

        # Get images that the model originally classified correctly
        correct_images = self.get_correct_images(
            max_images=max_images
        )

        if len(correct_images) == 0:

            raise RuntimeError(
                "No correctly classified test images were found."
            )

        successful_attacks = 0

        total_images = len(correct_images)

        # Attack every correctly classified image
        for image, label in correct_images:

            (
                adversarial_image,
                original_prediction,
                adversarial_prediction,
                attack_successful
            ) = self.attack_one_image(
                image,
                label,
                epsilon
            )

            if attack_successful:

                successful_attacks += 1

        # Calculate attack success rate
        success_rate = (
            100.0
            * successful_attacks
            / total_images
        )

        return (
            success_rate,
            successful_attacks,
            total_images
        )

    # =========================================================
    # 6. Experiment with different epsilon values
    # =========================================================

    def epsilon_experiment(
        self,
        epsilon_values,
        max_images=100
    ):

        results = []

        print()
        print("=" * 60)
        print("FGSM EPSILON EXPERIMENT")
        print("=" * 60)

        for epsilon in epsilon_values:

            (
                success_rate,
                successful_attacks,
                total_images
            ) = self.evaluate_epsilon(
                epsilon,
                max_images
            )

            print(
                f"Epsilon: {epsilon:.4f} | "
                f"Successful Attacks: "
                f"{successful_attacks}/{total_images} | "
                f"Success Rate: {success_rate:.2f}%"
            )

            results.append(
                {
                    "epsilon": epsilon,
                    "success_rate": success_rate,
                    "successful_attacks": successful_attacks,
                    "total_images": total_images
                }
            )

        return results

    # =========================================================
    # 7. Find minimum epsilon
    # =========================================================

    def find_minimum_epsilon(
        self,
        results,
        required_success_rate=50.0
    ):

        for result in results:

            if (
                result["success_rate"]
                >= required_success_rate
            ):

                print()
                print("=" * 60)
                print("MINIMUM EPSILON")
                print("=" * 60)

                print(
                    f"Minimum epsilon: "
                    f"{result['epsilon']:.4f}"
                )

                print(
                    f"Attack success rate: "
                    f"{result['success_rate']:.2f}%"
                )

                return result

        print()
        print(
            "No tested epsilon reached "
            f"{required_success_rate:.2f}% success rate."
        )

        return None

    # =========================================================
    # 8. Plot epsilon vs attack success rate
    # =========================================================

    def plot_epsilon_results(self, results):

        epsilons = [
            result["epsilon"]
            for result in results
        ]

        success_rates = [
            result["success_rate"]
            for result in results
        ]

        plt.figure(figsize=(8, 5))

        plt.plot(
            epsilons,
            success_rates,
            marker="o"
        )

        plt.xlabel("Epsilon")

        plt.ylabel("Attack Success Rate (%)")

        plt.title(
            "FGSM Attack Success Rate vs Epsilon"
        )

        plt.grid(True)

        plt.show()

    # =========================================================
    # 9. Generate one adversarial example
    # =========================================================

    def generate_example(
        self,
        epsilon
    ):

        correct_images = self.get_correct_images(
            max_images=1
        )

        if len(correct_images) == 0:

            raise RuntimeError(
                "No correctly classified image was found."
            )

        image, label = correct_images[0]

        (
            adversarial_image,
            original_prediction,
            adversarial_prediction,
            attack_successful
        ) = self.attack_one_image(
            image,
            label,
            epsilon
        )

        return (
            image,
            adversarial_image,
            label,
            original_prediction,
            adversarial_prediction,
            attack_successful
        )

    # =========================================================
    # 10. Visualize adversarial example
    # =========================================================

    def visualize_attack(
        self,
        image,
        adversarial_image,
        label,
        original_prediction,
        adversarial_prediction,
        epsilon
    ):

        # Remove batch dimension
        original = (
            image
            .squeeze(0)
            .detach()
            .cpu()
        )

        adversarial = (
            adversarial_image
            .squeeze(0)
            .detach()
            .cpu()
        )

        # Convert CHW -> HWC
        original = original.permute(
            1,
            2,
            0
        ).numpy()

        adversarial = adversarial.permute(
            1,
            2,
            0
        ).numpy()

        # Calculate perturbation
        perturbation = (
            adversarial
            - original
        )

        # For visualization only
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
        # Visualization
        # =====================================================

        plt.figure(figsize=(15, 5))

        # -----------------------------------------------------
        # Original image
        # -----------------------------------------------------

        plt.subplot(1, 3, 1)

        plt.imshow(original)

        plt.title(
            "Original Image\n"
            f"True: "
            f"{self.class_names[label.item()]}\n"
            f"Prediction: "
            f"{self.class_names[original_prediction.item()]}"
        )

        plt.axis("off")

        # -----------------------------------------------------
        # Adversarial image
        # -----------------------------------------------------

        plt.subplot(1, 3, 2)

        plt.imshow(adversarial)

        plt.title(
            "Adversarial Image\n"
            f"Prediction: "
            f"{self.class_names[adversarial_prediction.item()]}\n"
            f"Epsilon: {epsilon}"
        )

        plt.axis("off")

        # -----------------------------------------------------
        # Perturbation
        # -----------------------------------------------------

        plt.subplot(1, 3, 3)

        plt.imshow(
            perturbation_display
        )

        plt.title("Perturbation")

        plt.axis("off")

        plt.tight_layout()

        plt.show()

    # =============================================================
    # Run Phase 2
    # =============================================================

    def run(self, epsilons, max_images=100):

        print("\n" + "=" * 60)
        print("PHASE 2 - ADVERSARIAL ATTACK")
        print("=" * 60)

        results = {}

        for epsilon in epsilons:

            success_rate, successful_attacks, total_images = (
                self.evaluate_epsilon(
                    epsilon,
                    max_images=max_images
                )
            )

            adversarial_accuracy = 100.0 - success_rate
            results[epsilon] = adversarial_accuracy

            print(
                f"Epsilon: {epsilon:.4f} | "
                f"Successful Attacks: {successful_attacks}/{total_images} | "
                f"Attack Success Rate: {success_rate:.2f}% | "
                f"Adversarial Accuracy: {adversarial_accuracy:.2f}%"
            )

        print("\nPhase 2 completed successfully!")

        return results
