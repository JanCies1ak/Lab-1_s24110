import random
# train.py


class Point:
    def __init__(self, coordinates: tuple, cluster_number: int | None = None):
        self.coordinates = coordinates
        self.cluster_number = cluster_number

    def __getitem__(self, item: int):
        return self.coordinates[0]


class Model:
    def __init__(self):
        self.data: dict[int, list[Point]] | None = None

    def train(self, train_data: list[Point]):
        self.data = {}
        for point in train_data:
            if point.cluster_number not in self.data:
                self.data[point.cluster_number] = []

            self.data[point.cluster_number].append(point)

    def predict(self, predict_point: tuple | Point):
        result = None
        min_avg_distance = float('inf')

        for cluster_number, points in self.data.items():
            avg_distance = 0.0
            for point in points:
                avg_distance += ((point[0] - predict_point[0]) ** 2 + (point[0] - predict_point[0]) ** 2) ** 0.5
            avg_distance = avg_distance / len(points)

            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                result = cluster_number

        return result


# Generowanie prostego zbioru danych
def generate_data():
    amount_of_points_1 = random.randint(50, 100)
    amount_of_points_2 = random.randint(50, 100)

    data = []

    for i in range(amount_of_points_1):
        # Wartości w kwadracie ((0, 0) (1, 1))
        data.append(Point((random.random(), random.random()), 0))

    for i in range(amount_of_points_2):
        # Wartości w kwadracie ((1.5, 1) (2.5, 2))
        data.append(Point((random.random() + 1.5, random.random() + 1), 1))

    random.shuffle(data)

    return data


# Trenowanie prostego modelu regresji logistycznej
def train_model(model: Model, data: list[Point]):
    # Podział na zbiór treningowy i testowy

    train_threshold = int(len(data) * 0.8)
    train_data = data[:train_threshold]
    test_data = data[train_threshold:]

    # Trenowanie modelu

    model.train(train_data)

    # Predykcja na zbiorze testowym

    results = [model.predict(point) for point in test_data]

    # Wyliczenie dokładności

    accuracy = 0
    for i in range(len(test_data)):
        if results[i] == test_data[i].cluster_number:
            accuracy += 1

    accuracy /= len(test_data)

    # Zapis wyniku

    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")
    with open("accuracy.txt", "w") as file:
        file.write(f"Model trained with accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    train_model(Model(), generate_data())
