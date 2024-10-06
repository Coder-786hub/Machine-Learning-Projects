```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

```


```python
wine  = load_wine()
```


```python
dir(wine)
```




    ['DESCR', 'data', 'feature_names', 'frame', 'target', 'target_names']




```python
df = pd.DataFrame(wine.data , columns = wine.feature_names)
y = wine.target
```


```python
x
```




    array([[1.423e+01, 1.710e+00, 2.430e+00, ..., 1.040e+00, 3.920e+00,
            1.065e+03],
           [1.320e+01, 1.780e+00, 2.140e+00, ..., 1.050e+00, 3.400e+00,
            1.050e+03],
           [1.316e+01, 2.360e+00, 2.670e+00, ..., 1.030e+00, 3.170e+00,
            1.185e+03],
           ...,
           [1.179e+01, 2.130e+00, 2.780e+00, ..., 9.700e-01, 2.440e+00,
            4.660e+02],
           [1.237e+01, 1.630e+00, 2.300e+00, ..., 8.900e-01, 2.780e+00,
            3.420e+02],
           [1.204e+01, 4.300e+00, 2.380e+00, ..., 7.900e-01, 2.570e+00,
            5.800e+02]])




```python
y.shape
```




    (178,)




```python
x = x[y!=2]
y = y[y!=2]
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    Cell In[366], line 1
    ----> 1 x = x[y!=2]
          2 y = y[y!=2]
    

    IndexError: boolean index did not match indexed array along axis 0; size of axis is 130 but size of corresponding boolean axis is 178



```python
y
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2])




```python

```


```python

```


```python

```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])




```python
import tensorflow as tf   
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense
```


```python
from sklearn.preprocessing import StandardScaler
```


```python
from sklearn.model_selection import train_test_split
```


```python
x_train, x_test , y_train,y_test  = train_test_split(x,y,test_size = 0.2 , random_state = 42)
```


```python
scaler = StandardScaler()
```


```python
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
model = Sequential()
```


```python
x_train= x_train.reshape(104 , 13)
```


```python
x_train.shape
```




    (104, 13)




```python

```


```python
model.add(Dense(10, activation = 'relu' , input_shape = (13,)))
model.add(Dense(8, activation = 'tanh'))
model.add(Dense(1 , activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_corssentropy' ,metrics = ['accuracy'])

```


```python

# x_train = np.array(x_train)
# y_train = np.array(y_train)
# x_test = np.array(x_test)
# y_test = np.array(y_test)
```


```python
# print(x_train.shape)  
# print(y_train.shape)  
# print(x_test.shape)
# print(y_test.shape)
```


```python

```


```python
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[322], line 1
    ----> 1 model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\utils\traceback_utils.py:122, in filter_traceback.<locals>.error_handler(*args, **kwargs)
        119     filtered_tb = _process_traceback_frames(e.__traceback__)
        120     # To get the full stack trace, call:
        121     # `keras.config.disable_traceback_filtering()`
    --> 122     raise e.with_traceback(filtered_tb) from None
        123 finally:
        124     del filtered_tb
    

    File ~\AppData\Local\Programs\Python\Python312\Lib\site-packages\tensorflow\python\framework\constant_op.py:108, in convert_to_eager_tensor(value, ctx, dtype)
        106     dtype = dtypes.as_dtype(dtype).as_datatype_enum
        107 ctx.ensure_initialized()
    --> 108 return ops.EagerTensor(value, ctx.device_name, dtype)
    

    ValueError: object __array__ method not producing an array



```python

```


```python

```


```python

```


```python

```
