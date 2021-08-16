# Bondarevsky Test detector

This detector determines if a person is in the right position in one frame. 
More about the test [here](https://docs.google.com/document/d/1c548azsULPbLrI1ma2a4dj3d65e2qBNC2Gxu-cHru9s).

## Usage 
To run code, type in linux shell:
```
python3 src/main.py -i <path to Image>
```

## Example
```
python3 src/main.py -i examples/0.jpg
```
Output:
True

```
python3 src/main.py -i examples/1.jpg
```
Output:
(False, ('Left hand is not on belt', 'Right hand is not on belt'))



