
# 3D Printer DataSet for Mechanical Engineers

Import Data


```python
import pandas as pd
data = pd.read_csv("data.csv", sep = ";")
```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 50 entries, 0 to 49
    Data columns (total 12 columns):
    layer_height          50 non-null float64
    wall_thickness        50 non-null int64
    infill_density        50 non-null int64
    infill_pattern        50 non-null object
    nozzle_temperature    50 non-null int64
    bed_temperature       50 non-null int64
    print_speed           50 non-null int64
    material              50 non-null object
    fan_speed             50 non-null int64
    roughness             50 non-null int64
    tension_strenght      50 non-null int64
    elongation            50 non-null float64
    dtypes: float64(2), int64(8), object(2)
    memory usage: 4.8+ KB
    

Let's multiply these columns by 100 to make them more understandable.


```python
data.layer_height = data.layer_height*100
data.elongation = data.elongation*100
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>layer_height</th>
      <th>wall_thickness</th>
      <th>infill_density</th>
      <th>infill_pattern</th>
      <th>nozzle_temperature</th>
      <th>bed_temperature</th>
      <th>print_speed</th>
      <th>material</th>
      <th>fan_speed</th>
      <th>roughness</th>
      <th>tension_strenght</th>
      <th>elongation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>8</td>
      <td>90</td>
      <td>grid</td>
      <td>220</td>
      <td>60</td>
      <td>40</td>
      <td>abs</td>
      <td>0</td>
      <td>25</td>
      <td>18</td>
      <td>120.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>7</td>
      <td>90</td>
      <td>honeycomb</td>
      <td>225</td>
      <td>65</td>
      <td>40</td>
      <td>abs</td>
      <td>25</td>
      <td>32</td>
      <td>16</td>
      <td>140.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>1</td>
      <td>80</td>
      <td>grid</td>
      <td>230</td>
      <td>70</td>
      <td>40</td>
      <td>abs</td>
      <td>50</td>
      <td>40</td>
      <td>8</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>4</td>
      <td>70</td>
      <td>honeycomb</td>
      <td>240</td>
      <td>75</td>
      <td>40</td>
      <td>abs</td>
      <td>75</td>
      <td>68</td>
      <td>10</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>6</td>
      <td>90</td>
      <td>grid</td>
      <td>250</td>
      <td>80</td>
      <td>40</td>
      <td>abs</td>
      <td>100</td>
      <td>92</td>
      <td>5</td>
      <td>70.0</td>
    </tr>
  </tbody>
</table>
</div>



In this data set, ABS and PLA assigned 0 and 1 values for materials.


```python
data.material = [0 if each == "abs" else 1 for each in data.material]
# abs = 0, pla = 1

data.infill_pattern = [0 if each == "grid" else 1 for each in data.infill_pattern]
# grid = 0, honeycomb = 1
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>layer_height</th>
      <th>wall_thickness</th>
      <th>infill_density</th>
      <th>infill_pattern</th>
      <th>nozzle_temperature</th>
      <th>bed_temperature</th>
      <th>print_speed</th>
      <th>material</th>
      <th>fan_speed</th>
      <th>roughness</th>
      <th>tension_strenght</th>
      <th>elongation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>8</td>
      <td>90</td>
      <td>0</td>
      <td>220</td>
      <td>60</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>18</td>
      <td>120.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>7</td>
      <td>90</td>
      <td>1</td>
      <td>225</td>
      <td>65</td>
      <td>40</td>
      <td>0</td>
      <td>25</td>
      <td>32</td>
      <td>16</td>
      <td>140.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>230</td>
      <td>70</td>
      <td>40</td>
      <td>0</td>
      <td>50</td>
      <td>40</td>
      <td>8</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>4</td>
      <td>70</td>
      <td>1</td>
      <td>240</td>
      <td>75</td>
      <td>40</td>
      <td>0</td>
      <td>75</td>
      <td>68</td>
      <td>10</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>6</td>
      <td>90</td>
      <td>0</td>
      <td>250</td>
      <td>80</td>
      <td>40</td>
      <td>0</td>
      <td>100</td>
      <td>92</td>
      <td>5</td>
      <td>70.0</td>
    </tr>
  </tbody>
</table>
</div>



Seperate Input parameters and Prediction Materials.


```python
y_data = data.material.values
x_data = data.drop(["material"],axis=1)
```


```python
absm = data[data.material == 0]
pla = data[data.material == 1]
```


```python
absm.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>layer_height</th>
      <th>wall_thickness</th>
      <th>infill_density</th>
      <th>infill_pattern</th>
      <th>nozzle_temperature</th>
      <th>bed_temperature</th>
      <th>print_speed</th>
      <th>material</th>
      <th>fan_speed</th>
      <th>roughness</th>
      <th>tension_strenght</th>
      <th>elongation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>8</td>
      <td>90</td>
      <td>0</td>
      <td>220</td>
      <td>60</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>18</td>
      <td>120.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>7</td>
      <td>90</td>
      <td>1</td>
      <td>225</td>
      <td>65</td>
      <td>40</td>
      <td>0</td>
      <td>25</td>
      <td>32</td>
      <td>16</td>
      <td>140.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>1</td>
      <td>80</td>
      <td>0</td>
      <td>230</td>
      <td>70</td>
      <td>40</td>
      <td>0</td>
      <td>50</td>
      <td>40</td>
      <td>8</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>4</td>
      <td>70</td>
      <td>1</td>
      <td>240</td>
      <td>75</td>
      <td>40</td>
      <td>0</td>
      <td>75</td>
      <td>68</td>
      <td>10</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>6</td>
      <td>90</td>
      <td>0</td>
      <td>250</td>
      <td>80</td>
      <td>40</td>
      <td>0</td>
      <td>100</td>
      <td>92</td>
      <td>5</td>
      <td>70.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
```


```python
plt.scatter(absm.fan_speed,absm.tension_strenght,color="red",label="ABS",alpha= 0.5)
plt.scatter(pla.fan_speed,pla.tension_strenght,color="green",label="PLA",alpha= 0.5)
plt.xlabel("Fan Speed")
plt.ylabel("Tension Strength")
plt.legend()
plt.show()
```


![png](output_15_0.png)


As you see, the air circulation not good for ABS


```python
plt.scatter(absm.layer_height,absm.roughness,color="blue",label="ABS",alpha= 0.9)
plt.scatter(pla.layer_height,pla.roughness,color="pink",label="PLA",alpha= 0.9)
plt.xlabel("Layer Height")
plt.ylabel("Roughness")
plt.legend()
plt.show()
```


![png](output_17_0.png)


You can see as the layer height increases, the tensile strength increases. But PLA smoother than ABS


```python
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = data.infill_density
y = data.wall_thickness
z = data.tension_strenght

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('Infill Density')
ax.set_ylabel('Wall Thickness')
ax.set_zlabel('Tension Strenght')

plt.show()
```


![png](output_19_0.png)



```python
# normalization 
x_norm = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_norm,y_data,test_size = 0.3,random_state=1)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))

score_list = []
for each in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    print(" {} nn score: {} ".format(each,knn2.score(x_test,y_test)))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
```

     3 nn score: 0.6 
     1 nn score: 0.4666666666666667 
     2 nn score: 0.4666666666666667 
     3 nn score: 0.6 
     4 nn score: 0.6666666666666666 
     5 nn score: 0.7333333333333333 
     6 nn score: 0.6666666666666666 
     7 nn score: 0.7333333333333333 
     8 nn score: 0.6 
     9 nn score: 0.7333333333333333 
     10 nn score: 0.6 
     11 nn score: 0.6 
     12 nn score: 0.6666666666666666 
     13 nn score: 0.7333333333333333 
     14 nn score: 0.6 
    


![png](output_20_1.png)



```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Input, Dense, Flatten
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(Dense(32,input_dim=11))
model.add(BatchNormalization(axis = -1))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(16))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_data,y, epochs=500, batch_size =32, validation_split= 0.20)
```


```python
a1 = 4 #layer_height*100
a2 = 5 #wall_thickness
a3 = 60 #infill_density
a4 = 0 #infilkk_pattern
a5 = 232 #nozzle_temperature 
a6 = 74 #bed_temperature
a7 = 90 #print_speed
a8 = 100 #fan_speed
a9 = 150 #roughness
a10 = 30 #tension_strenght
a11 = 200 #elangation*100

tahmin = np.array([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11]).reshape(1,11)
print(model.predict_classes(tahmin))

if model.predict_classes(tahmin) == 0: 
    print("Material is ABS")
else:   
    print("Material is PLA.")
```
