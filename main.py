import pandas as pd
from MLP import MultilayerBackpropagation, calculate_confusion_matrix, calculate_confusion_matrix_dynamic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def run(selected_features, selected_classes, learning_rate, num_of_epochs, bias, layers, activation_function):
    df = pd.read_csv('Dry_Bean_Dataset.csv')

    mean_minor_axis_length = df['MinorAxisLength'].mean()
    df['MinorAxisLength'].fillna(mean_minor_axis_length, inplace=True)

    selected_df = df[df['Class'].isin(selected_classes)]
    area_normalizer = MinMaxScaler()
    perimeter_normalizer = MinMaxScaler()
    major_normalizer = MinMaxScaler()
    minor_normalizer = MinMaxScaler()
    selected_df.loc[:, 'Area'] = area_normalizer.fit_transform(selected_df[['Area']])
    selected_df.loc[:, 'Perimeter'] = perimeter_normalizer.fit_transform(selected_df[['Perimeter']])
    selected_df.loc[:, 'MajorAxisLength'] = major_normalizer.fit_transform(selected_df[['MajorAxisLength']])
    selected_df.loc[:, 'MinorAxisLength'] = minor_normalizer.fit_transform(selected_df[['MinorAxisLength']])
    x_train, x_test, y_train, y_test = train_test_split(
        selected_df[selected_features], selected_df['Class'],
        test_size=0.4, random_state=0, shuffle=True
    )
    le = LabelEncoder()

    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    mlb = MultilayerBackpropagation(bias, learning_rate, num_of_epochs, layers, activation_function)
    mlb.train(x_train, y_train)
    y_predict = mlb.predict(x_test)
    print("Train")
    mlb.evaluate(x_train, y_train)
    print("Test")
    mlb.evaluate(x_test, y_test)
    if len(selected_classes) < 3:
        calculate_confusion_matrix_dynamic(y_test, y_predict)
    else:
        calculate_confusion_matrix(y_test, y_predict)


