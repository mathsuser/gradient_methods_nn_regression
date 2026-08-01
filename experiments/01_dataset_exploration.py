from gradient_methods_nn_regression.data import generate_synthetic_regression_data


def main() -> None:
    x, y_noisy, y_true, noise = generate_synthetic_regression_data(
        n_samples=1000,
        n_features=8,
        noise_std=0.2,
        seed=123,
    )

    print("x shape:", x.shape)
    print("y_noisy shape:", y_noisy.shape)
    print("y_true shape:", y_true.shape)
    print("noise shape:", noise.shape)


if __name__ == "__main__":
    main()