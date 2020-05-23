def show_head(cars):
    print(cars.head())


def show_norm_loss(cars):
    print("normalized losses: ", cars['normalized_losses'].isnull().sum())

