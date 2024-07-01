import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

col_names = np.array(["duration", "protocol_type", "service", "flag", "src_bytes",
                      "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                      "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                      "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                      "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                      "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                      "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                      "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                      "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                      "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "levels"])

print('Feature names: ', col_names)

# loading data
train_data = pd.read_csv("NSL-KDD-train.txt", names=col_names)
test_data = pd.read_csv("NSL-KDD-test.txt", names=col_names)

print('Dimensions of the training data: ', train_data.shape)
print('Dimensions of the test data: ', test_data.shape)

print(train_data.head(5))

# DATA PREPROCESSING

# check for null data
print('Null Values: \n', train_data.isnull().sum())

# check for duplicate data
print('Duplicate Count: ', train_data.duplicated().sum())

categorical_features = np.array(["protocol_type", "service", "flag", "attack"])

# create a new column to map 'normal' attacks as value 0 and anything else to 1
train_attack = train_data.attack.map(lambda a: 0 if a == 'normal' else 1)
test_attack = test_data.attack.map(lambda a: 0 if a == 'normal' else 1)

# add new column to dataframe
train_data['attack_flag'] = train_attack
test_data['attack_flag'] = test_attack

print(train_data.head(5))


def print_unique_values(df, columns):  # method to find unique values and their counts in categorical features
    for column_name in columns:
        print('Column: ', column_name, '\n')
        unique_vals = df[column_name].unique()
        value_counts = df[column_name].value_counts()
        print('Unique Values: ', unique_vals, '\n')
        print('Counts: ', value_counts, '\n')


# check for unique values in categorical features
print_unique_values(train_data, categorical_features)

# create lists to classify attack types
dos_attacks = np.array(['neptune', 'smurf', 'back', 'teardrop', 'pod', 'land', ])
probe_attacks = np.array(['satan', 'ipsweep', 'portsweep', 'nmap', ])
privilege_attacks = np.array(['buffer_overflow', 'rootkit', 'loadmodule', 'perl'])
access_attacks = np.array(['warezclient', 'guess_passwd', 'warezmaster', 'imap', 'ftp_write', 'multihop', 'phf', 'spy'])

attack_labels = np.array(['normal', 'dos', 'probe', 'privilege', 'access'])


def map_attack_type(attack):  # method to map attack types into numeric values
    if attack in dos_attacks:
        attack_type = 1
    elif attack in probe_attacks:
        attack_type = 2
    elif attack in privilege_attacks:
        attack_type = 3
    elif attack in access_attacks:
        attack_type = 4
    else:
        attack_type = 0

    return attack_type


# create new column for attack type and add to dataframe
train_attack_type = train_data.attack.apply(map_attack_type)
train_data['attack_type'] = train_attack_type

test_attack_type = test_data.attack.apply(map_attack_type)
test_data['attack_type'] = test_attack_type

print(train_data.head())

# Explanatory Data Analysis

only_attacks = train_data[train_data.attack_flag == 1]
only_normal = train_data[train_data.attack_flag == 0]

# attacks by protocol type
plt.figure(figsize=(5, 5))
sns.countplot(x='attack_flag', data=train_data, hue='protocol_type')
plt.xticks(rotation=45)
plt.title('Attacks by Protocol Type', fontdict={'fontsize': 16})
plt.show()

protocol_counts_attack = only_attacks['protocol_type'].value_counts().sort_values(ascending=False)
protocol_counts_normal = only_normal['protocol_type'].value_counts().sort_values(ascending=False)

# shown in pie-chart for percentage representation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.pie(protocol_counts_attack, labels=protocol_counts_attack.index, autopct="%1.1f%%")
ax1.set_title("Attacks")
ax2.pie(protocol_counts_normal, labels=protocol_counts_normal.index, autopct="%1.1f%%")
ax2.set_title("Non-Attacks")
fig.suptitle("Protocol Distribution by Attack")
plt.tight_layout()
plt.show()

# attack types by protocol type
plt.figure(figsize=(10, 5))
sns.countplot(x='attack', data=train_data, hue='protocol_type')
plt.xticks(rotation=45)
plt.title('Attack Types by Protocol Type', fontdict={'fontsize': 16})
plt.show()

# attacks by service used
plt.figure(figsize=(30, 5))
sns.countplot(x='service', hue='attack_flag', data=train_data)
plt.xticks(rotation=45)
plt.title('Attacks by Service Used')
plt.show()

# attacks by flag
plt.figure(figsize=(10, 5))
sns.countplot(x='flag', hue='attack_flag', data=train_data)
plt.xticks(rotation=45)
plt.title('Attacks by Flag')
plt.show()

flag_counts_attack = only_attacks['flag'].value_counts().sort_values(ascending=False)
flag_counts_normal = only_normal['flag'].value_counts().sort_values(ascending=False)

# shown in pie-chart for percentage representation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.pie(flag_counts_attack, labels=flag_counts_attack.index, autopct="%1.1f%%")
ax1.set_title("Attacks")
ax2.pie(flag_counts_normal, labels=flag_counts_normal.index, autopct="%1.1f%%")
ax2.set_title("Non-Attacks")
fig.suptitle("Flag Distribution by Attack")
plt.tight_layout()
plt.show()

# Apply Data Mining

# encode categorical features to get numeric values
label_encoder = preprocessing.LabelEncoder()
for feature in categorical_features[:-1]:  # skip attack as it has already been processed
    train_data[feature] = label_encoder.fit_transform(train_data[feature])
    test_data[feature] = label_encoder.fit_transform(test_data[feature])

selected_numeric_features = np.array(['duration', 'src_bytes', 'dst_bytes', 'num_failed_logins'])

fit_set = train_data[categorical_features[:-1]]
fit_set = fit_set.join(train_data[selected_numeric_features])

test_set = test_data[categorical_features[:-1]]
test_set = test_set.join(train_data[selected_numeric_features])

y_flag = train_data['attack_flag']
y_mapped = train_data['attack_type']

y_test_flag = test_data['attack_flag']
y_test_mapped = test_data['attack_type']

X_train_flag, X_val_flag, y_train_flag, y_val_flag = train_test_split(fit_set, y_flag, test_size=0.6)
X_train_mapped, X_val_mapped, y_train_mapped, y_val_mapped = train_test_split(fit_set, y_mapped, test_size=0.6)

"""

# list models for testing
models = [
    RandomForestClassifier(),
    KNeighborsClassifier(),
    svm.SVC(kernel='rbf'),
    GradientBoostingClassifier()
]

model_comps = []

# calculate accuracy for attack flag
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, X_train_flag, y_train_flag, scoring='accuracy')
    for count, accuracy in enumerate(accuracies):
        model_comps.append((model_name, count, accuracy))

# put the results in a dataframe and plot it for comparison
result_df = pd.DataFrame(model_comps, columns=['model_name', 'count', 'accuracy'])

pivot_table = result_df.pivot_table(index='count', columns='model_name', values='accuracy')
plt.figure(figsize=(10, 10))
pivot_table.boxplot(rot=45)
plt.title('Model Comparison - Attack Flag')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)

plt.show()

# calculate accuracy for attack map
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, X_train_mapped, y_train_mapped, scoring='accuracy')
    for count, accuracy in enumerate(accuracies):
        model_comps.append((model_name, count, accuracy))

# put the results in a dataframe and plot it for comparison
result_df = pd.DataFrame(model_comps, columns=['model_name', 'count', 'accuracy'])

pivot_table = result_df.pivot_table(index='count', columns='model_name', values='accuracy')
plt.figure(figsize=(10, 10))
pivot_table.boxplot(rot=45)
plt.title('Model Comparison - Attack Map')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)

plt.show()

"""

# best values are with KNeighbors and RandomForest classifiers so use them

# Random Forest Classifier for Attack Flag
rfc = RandomForestClassifier()
rfc.fit(fit_set, y_flag)
rfc_prediction = rfc.predict(test_set)

rfc_score = accuracy_score(y_test_flag, rfc_prediction)

# Random Forest Classifier for Attack Types
rfc_mapped = RandomForestClassifier()
rfc_mapped.fit(fit_set, y_mapped)
rfc_mapped_prediction = rfc_mapped.predict(test_set)

rfc_score_mapped = accuracy_score(y_test_mapped, rfc_prediction)

print('Attack Flag: ', rfc_score, 'Attack Type: ', rfc_score_mapped)

# KNeighbors Classifier for Attack Flag
knc = KNeighborsClassifier()
knc.fit(fit_set, y_flag)
knc_prediction = knc.predict(test_set)

knc_score = accuracy_score(y_test_flag, knc_prediction)

# KNeighbors Classifier for Attack Types
knc_mapped = KNeighborsClassifier()
knc_mapped.fit(fit_set, y_mapped)
knc_mapped_prediction = knc_mapped.predict(test_set)

knc_score_mapped = accuracy_score(y_test_mapped, knc_mapped_prediction)

print('Attack Flag: ', knc_score, 'Attack Type: ', knc_score_mapped)

