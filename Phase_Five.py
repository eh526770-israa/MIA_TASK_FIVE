import copy

import torch
import torch.nn as nn
import torch.optim as optim


class Phase5:

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        criterion,
        device
    ):

        # =====================================================
        # Existing objects from Phase 1
        # =====================================================

        self.original_model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device

        # =====================================================
        # Create a separate copy for adversarial training
        # =====================================================

        self.hardened_model = copy.deepcopy(
            self.original_model
        )

        self.hardened_model = self.hardened_model.to(
            self.device
        )

        # =====================================================
        # Make the same layers trainable as Phase 1
        #
        # Phase 1:
        # - All parameters frozen
        # - layer4 trainable
        # - fc trainable
        # =====================================================

        for param in self.hardened_model.parameters():

            param.requires_grad = False

        for param in self.hardened_model.layer4.parameters():

            param.requires_grad = True

        for param in self.hardened_model.fc.parameters():

            param.requires_grad = True

    # =========================================================
    # FGSM Attack
    # =========================================================

    def fgsm_attack(
        self,
        images,
        labels,
        epsilon
    ):

        # -----------------------------------------------------
        # We need gradients with respect to the input image
        # -----------------------------------------------------

        images = images.clone().detach()

        images.requires_grad_(True)

        # -----------------------------------------------------
        # Forward pass
        # -----------------------------------------------------

        outputs = self.hardened_model(
            images
        )

        # -----------------------------------------------------
        # Calculate loss
        # -----------------------------------------------------

        loss = self.criterion(
            outputs,
            labels
        )

        # -----------------------------------------------------
        # Clear old gradients
        # -----------------------------------------------------

        self.hardened_model.zero_grad()

        # -----------------------------------------------------
        # Calculate gradient of loss
        # with respect to input image
        # -----------------------------------------------------

        loss.backward()

        image_gradient = images.grad.detach()

        # -----------------------------------------------------
        # FGSM perturbation
        # -----------------------------------------------------

        perturbation = (
            epsilon
            * image_gradient.sign()
        )

        # -----------------------------------------------------
        # Create adversarial image
        # -----------------------------------------------------

        adversarial_images = (
            images.detach()
            + perturbation
        )

        # -----------------------------------------------------
        # Keep pixel values in valid range [0, 1]
        # -----------------------------------------------------

        adversarial_images = torch.clamp(
            adversarial_images,
            0.0,
            1.0
        )

        return adversarial_images.detach()

    # =========================================================
    # Train one epoch
    # =========================================================

    def train_one_epoch(
        self,
        optimizer,
        epsilon
    ):

        self.hardened_model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for images, labels in self.train_loader:

            images = images.to(
                self.device
            )

            labels = labels.to(
                self.device
            )

            # =================================================
            # Generate adversarial examples
            # =================================================

            adversarial_images = self.fgsm_attack(
                images,
                labels,
                epsilon
            )

            # =================================================
            # Clear optimizer gradients
            # =================================================

            optimizer.zero_grad()

            # =================================================
            # Train on CLEAN images
            # =================================================

            clean_outputs = self.hardened_model(
                images
            )

            clean_loss = self.criterion(
                clean_outputs,
                labels
            )

            # =================================================
            # Train on ADVERSARIAL images
            # =================================================

            adversarial_outputs = self.hardened_model(
                adversarial_images
            )

            adversarial_loss = self.criterion(
                adversarial_outputs,
                labels
            )

            # =================================================
            # Combined loss
            # =================================================

            loss = (
                0.5 * clean_loss
                +
                0.5 * adversarial_loss
            )

            # =================================================
            # Backpropagation
            # =================================================

            loss.backward()

            optimizer.step()

            # =================================================
            # Statistics on clean images
            # =================================================

            running_loss += loss.item()

            predictions = torch.argmax(
                clean_outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

        epoch_loss = (
            running_loss
            / len(self.train_loader)
        )

        epoch_accuracy = (
            100.0
            * correct
            / total
        )

        return epoch_loss, epoch_accuracy

    # =========================================================
    # Evaluate model
    # =========================================================

    def evaluate(
        self,
        model,
        loader
    ):

        model.eval()

        running_loss = 0.0

        correct = 0

        total = 0

        with torch.no_grad():

            for images, labels in loader:

                images = images.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )

                outputs = model(
                    images
                )

                loss = self.criterion(
                    outputs,
                    labels
                )

                running_loss += loss.item()

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                total += labels.size(0)

                correct += (
                    predictions == labels
                ).sum().item()

        loss = (
            running_loss
            / len(loader)
        )

        accuracy = (
            100.0
            * correct
            / total
        )

        return loss, accuracy

    # =========================================================
    # Evaluate model on adversarial test set
    # =========================================================

    def evaluate_adversarial(
        self,
        model,
        epsilon
    ):

        model.eval()

        correct = 0

        total = 0

        for images, labels in self.test_loader:

            images = images.to(
                self.device
            )

            labels = labels.to(
                self.device
            )

            # =================================================
            # Generate adversarial images
            # =================================================

            #
            # Important:
            # FGSM needs gradients.
            #
            # Therefore we CANNOT use torch.no_grad()
            # while generating the attack.
            #

            images_for_attack = (
                images.clone().detach()
            )

            images_for_attack.requires_grad_(
                True
            )

            outputs = model(
                images_for_attack
            )

            loss = self.criterion(
                outputs,
                labels
            )

            model.zero_grad()

            loss.backward()

            gradients = (
                images_for_attack.grad.detach()
            )

            adversarial_images = (
                images_for_attack.detach()
                +
                epsilon * gradients.sign()
            )

            adversarial_images = torch.clamp(
                adversarial_images,
                0.0,
                1.0
            )

            # =================================================
            # Evaluate adversarial images
            # =================================================

            with torch.no_grad():

                adversarial_outputs = model(
                    adversarial_images
                )

                predictions = torch.argmax(
                    adversarial_outputs,
                    dim=1
                )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

        accuracy = (
            100.0
            * correct
            / total
        )

        return accuracy

    # =========================================================
    # Run adversarial training
    # =========================================================

    def train(
        self,
        epsilon,
        epochs=5,
        learning_rate=0.0001
    ):

        print()
        print("=" * 70)
        print("PHASE 5 - ADVERSARIAL TRAINING")
        print("=" * 70)

        print(
            "Defense:",
            "Adversarial Training"
        )

        print(
            "Epsilon:",
            epsilon
        )

        print(
            "Epochs:",
            epochs
        )

        # =====================================================
        # Optimizer for HARDENED model only
        # =====================================================

        optimizer = optim.Adam(

            filter(
                lambda p: p.requires_grad,
                self.hardened_model.parameters()
            ),

            lr=learning_rate
        )

        # =====================================================
        # Training
        # =====================================================

        for epoch in range(epochs):

            train_loss, train_accuracy = (
                self.train_one_epoch(
                    optimizer,
                    epsilon
                )
            )

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"| Loss: {train_loss:.4f} "
                f"| Clean Train Acc: "
                f"{train_accuracy:.2f}%"
            )

        print()
        print(
            "Adversarial training completed."
        )

        return self.hardened_model

    # =========================================================
    # Final Showdown
    # =========================================================

    def showdown(
        self,
        epsilon
    ):

        print()
        print("=" * 70)
        print("PHASE 5 - FINAL SHOWDOWN")
        print("=" * 70)

        # =====================================================
        # ORIGINAL MODEL - CLEAN TEST
        # =====================================================

        original_clean_loss, original_clean_accuracy = (
            self.evaluate(
                self.original_model,
                self.test_loader
            )
        )

        # =====================================================
        # ORIGINAL MODEL - ADVERSARIAL TEST
        # =====================================================

        original_adversarial_accuracy = (
            self.evaluate_adversarial(
                self.original_model,
                epsilon
            )
        )

        # =====================================================
        # HARDENED MODEL - CLEAN TEST
        # =====================================================

        hardened_clean_loss, hardened_clean_accuracy = (
            self.evaluate(
                self.hardened_model,
                self.test_loader
            )
        )

        # =====================================================
        # HARDENED MODEL - ADVERSARIAL TEST
        # =====================================================

        hardened_adversarial_accuracy = (
            self.evaluate_adversarial(
                self.hardened_model,
                epsilon
            )
        )

        # =====================================================
        # Print results
        # =====================================================

        print()
        print(
            "Original Model - Clean Accuracy:",
            f"{original_clean_accuracy:.2f}%"
        )

        print(
            "Original Model - Adversarial Accuracy:",
            f"{original_adversarial_accuracy:.2f}%"
        )

        print()

        print(
            "Hardened Model - Clean Accuracy:",
            f"{hardened_clean_accuracy:.2f}%"
        )

        print(
            "Hardened Model - Adversarial Accuracy:",
            f"{hardened_adversarial_accuracy:.2f}%"
        )

        # =====================================================
        # Calculate improvements
        # =====================================================

        robustness_improvement = (
            hardened_adversarial_accuracy
            - original_adversarial_accuracy
        )

        clean_accuracy_change = (
            hardened_clean_accuracy
            - original_clean_accuracy
        )

        print()
        print(
            "Adversarial Accuracy Improvement:",
            f"{robustness_improvement:+.2f}%"
        )

        print(
            "Clean Accuracy Change:",
            f"{clean_accuracy_change:+.2f}%"
        )

        # =====================================================
        # Save results
        # =====================================================

        results = {

            "original_clean_accuracy":
                original_clean_accuracy,

            "original_adversarial_accuracy":
                original_adversarial_accuracy,

            "hardened_clean_accuracy":
                hardened_clean_accuracy,

            "hardened_adversarial_accuracy":
                hardened_adversarial_accuracy,

            "adversarial_accuracy_improvement":
                robustness_improvement,

            "clean_accuracy_change":
                clean_accuracy_change
        }

        return results

    # =========================================================
    # Save hardened model
    # =========================================================

    def save_model(
        self,
        path="resnet34_caltech101_phase5_hardened.pth"
    ):

        torch.save(
            self.hardened_model.state_dict(),
            path
        )

        print()
        print(
            "Hardened model saved successfully:"
        )

        print(path)

