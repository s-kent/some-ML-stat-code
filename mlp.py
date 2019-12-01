base_dir = '../input/';

# load features, ignore header, ignore IDs
X_train1 = np.loadtxt(base_dir + 'train_features.csv', delimiter=',')[:,1:];
X_test1 = np.loadtxt(base_dir + 'test_features.csv', delimiter=',')[:,1:];
y_train1 = np.loadtxt(base_dir + 'train_labels.csv', dtype=np.uint8, delimiter=',', skiprows=1)[:,-1];


im_train = X_train1[0,:].reshape((30,30,3), order='F')
im_test = X_test1[0,:].reshape((30,30,3), order='F')

plt.figure(1)
plt.imshow(im_train/255)
plt.axis('off')

plt.figure(2)
plt.imshow(im_test/255)
plt.axis('off');

from sklearn import preprocessing
#pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)
#X_train = pt.fit_transform(X_train) 
X_train1 = preprocessing.scale(X_train1)

print('ok')

X_train, X_test, y_train, y_test = train_test_split( X_train1, y_train1, test_size=0.3, random_state=13)
from sklearn import svm 
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
print(X_train.shape)
clf6 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(2000, 120), random_state=1)

#clf4 = svm.SVC(gamma=0.00001,kernel='linear',max_iter=5000,coef0=0.01,degree=3,C=1,probability=True)
#clf5 = linear_model.SGDClassifier(max_iter=10000)
#clf6 = RandomForestClassifier(n_estimators=100, max_depth=8,random_state=0,min_impurity_decrease=0.0002,criterion='gini',n_jobs=-1)
clf6.fit(X_train, y_train)

y_pred = clf6.predict(X_train)
y_pred2 = clf6.predict(X_test)
print("acc: ",accuracy_score(y_train, y_pred))
print("acc: ",accuracy_score(y_test, y_pred2))
#print(np.mean(y_pred==y_test))