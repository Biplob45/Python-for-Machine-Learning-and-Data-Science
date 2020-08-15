```python
#!pip install tensorflow
#!pip install keras
```


```python
import tensorflow 
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
```


```python
sentences = [
    'i love my cat',
    'i love my dog',
    'You love my ,,,dog?',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
```

    {'my': 1, 'love': 2, 'dog': 3, 'i': 4, 'you': 5, 'cat': 6, 'do': 7, 'think': 8, 'is': 9, 'amazing': 10}



```python
sentences = [
    'i love my cat',
    'i love my dog',
    'You love my ,,,dog?',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
```

    [[4, 2, 1, 6], [4, 2, 1, 3], [5, 2, 1, 3], [7, 5, 8, 1, 3, 9, 10]]



```python
test_data = [
    'I really love my dog',
    'You suck'
]

sequences = tokenizer.texts_to_sequences(test_data)
print(sequences)
```

    [[4, 2, 1, 3], [5]]



```python
sentences = [
    'i love my cat',
    'i love my dog',
    'You love my ,,,dog?',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words=100,oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

test_data = [
    'I really love my dog',
    'You suck'
]

sequences = tokenizer.texts_to_sequences(test_data)
print(sequences)
```

    {'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}
    [[5, 1, 3, 2, 4], [6, 1]]



```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'i love my cat',
    'i love my dog',
    'You love my ,,,dog?',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words=100,oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

test_data = [
    'I really love my dog',
    'You suck'
]

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences)
print(sequences)
print(padded)
```

    {'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}
    [[5, 3, 2, 7], [5, 3, 2, 4], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]
    [[ 0  0  0  5  3  2  7]
     [ 0  0  0  5  3  2  4]
     [ 0  0  0  6  3  2  4]
     [ 8  6  9  2  4 10 11]]



```python

```
