# =========================================================
# IMPORT PHASES
# =========================================================

from Phase_One import Phase1
from Phase_Two import Phase2
from Phase_Three import Phase3
from Phase_Four import Phase4
from Phase_Five import Phase5


def main():

    # =====================================================
    # PHASE 1
    # Dataset + Model + Training + Evaluation
    # =====================================================

    print("\n")
    print("=" * 70)
    print("STARTING PHASE 1")
    print("=" * 70)

    phase1 = Phase1()

    phase1_results = phase1.run(
        num_epochs=5
    )

    # -----------------------------------------------------
    # Get objects created by Phase 1
    # -----------------------------------------------------

    model = phase1_results["model"]

    dataset = phase1_results["dataset"]

    train_loader = phase1_results["train_loader"]

    val_loader = phase1_results["val_loader"]

    test_loader = phase1_results["test_loader"]

    criterion = phase1_results["criterion"]

    device = phase1_results["device"]

    phase1_accuracy = phase1_results["test_accuracy"]

    print("\nPhase 1 completed.")
    print(
        f"Phase 1 Test Accuracy: "
        f"{phase1_accuracy:.2f}%"
    )


    # =====================================================
    # PHASE 2
    # Adversarial Attack
    # =====================================================

    print("\n")
    print("=" * 70)
    print("STARTING PHASE 2")
    print("=" * 70)

    phase2 = Phase2(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    # -----------------------------------------------------
    # Test different epsilon values
    # -----------------------------------------------------

    epsilons = [
        0.001,
        0.005,
        0.01,
        0.02
    ]

    phase2_results = phase2.run(
        epsilons=epsilons
    )

    print("\nPhase 2 completed.")


    # -----------------------------------------------------
    # Find strongest attack
    #
    # Lower adversarial accuracy means
    # stronger attack.
    # -----------------------------------------------------

    best_epsilon = min(
        phase2_results,
        key=phase2_results.get
    )

    best_attack_accuracy = (
        phase2_results[best_epsilon]
    )

    print(
        f"\nSelected epsilon: "
        f"{best_epsilon}"
    )

    print(
        f"Adversarial Accuracy: "
        f"{best_attack_accuracy:.2f}%"
    )


    # =====================================================
    # PHASE 3
    # Model Explainability
    # =====================================================

    print("\n")
    print("=" * 70)
    print("STARTING PHASE 3")
    print("=" * 70)

    phase3 = Phase3(
        model=model,
        test_loader=test_loader,
        class_names=dataset.categories,
        device=device
    )

    phase3_results = phase3.run()

    print("\nPhase 3 completed.")


    # =====================================================
    # PHASE 4
    # Forensic Analysis
    # =====================================================

    print("\n")
    print("=" * 70)
    print("STARTING PHASE 4")
    print("=" * 70)

    phase4 = Phase4(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        phase2=phase2,
        phase3=phase3,
        class_names=dataset.categories,
        device=device
    )

    phase4_results = phase4.run(
        epsilon=best_epsilon
    )

    print("\nPhase 4 completed.")


    # =====================================================
    # PHASE 5
    # Adversarial Training / Defense
    # =====================================================

    print("\n")
    print("=" * 70)
    print("STARTING PHASE 5")
    print("=" * 70)

    phase5 = Phase5(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    # -----------------------------------------------------
    # Train hardened model
    # -----------------------------------------------------

    hardened_model = phase5.train(
        epsilon=best_epsilon,
        epochs=5,
        learning_rate=0.0001
    )

    # -----------------------------------------------------
    # Final comparison
    # -----------------------------------------------------

    phase5_results = phase5.showdown(
        epsilon=best_epsilon
    )

    # -----------------------------------------------------
    # Save hardened model
    # -----------------------------------------------------

    phase5.save_model()

    print("\nPhase 5 completed.")


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("\n")
    print("=" * 70)
    print("ALL PHASES COMPLETED")
    print("=" * 70)

    print(
        f"\nPhase 1 Clean Accuracy: "
        f"{phase1_accuracy:.2f}%"
    )

    print(
        f"Phase 2 Best Epsilon: "
        f"{best_epsilon}"
    )

    print(
        f"Phase 2 Adversarial Accuracy: "
        f"{best_attack_accuracy:.2f}%"
    )

    print("\nPhase 3:")
    print("Saliency Maps + Grad-CAM")

    print("\nPhase 4:")
    print("Forensic Analysis completed.")

    print("\nPhase 5:")

    print(
        f"Original Clean Accuracy: "
        f"{phase5_results['original_clean_accuracy']:.2f}%"
    )

    print(
        f"Original Adversarial Accuracy: "
        f"{phase5_results['original_adversarial_accuracy']:.2f}%"
    )

    print(
        f"Hardened Clean Accuracy: "
        f"{phase5_results['hardened_clean_accuracy']:.2f}%"
    )

    print(
        f"Hardened Adversarial Accuracy: "
        f"{phase5_results['hardened_adversarial_accuracy']:.2f}%"
    )

    print(
        f"Adversarial Accuracy Improvement: "
        f"{phase5_results['adversarial_accuracy_improvement']:+.2f}%"
    )

    print(
        f"Clean Accuracy Change: "
        f"{phase5_results['clean_accuracy_change']:+.2f}%"
    )

    print("\n")
    print("=" * 70)
    print("PROJECT FINISHED SUCCESSFULLY")
    print("=" * 70)


# =========================================================
# RUN MAIN
# =========================================================

if __name__ == "__main__":
    main()

