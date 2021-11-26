# API for ML model

## what is API?
* It is a program that receives data in the form of request from user, process it and send other data in the form of response from server.
* It's an access point.
###Example
Google search - client send request by providing data,
process it by finding relevant data and send that data to the client.

## what is ML?
Machine Learning
* ML is a sort of data analysis where machine learn from data, identify patterns and predict or make decisions accordingly.

### Example
E-commerce - websites like flipkart and amazon use ML algorithm or ML model to provide relevant data to client.

## what is HTTP?
Hyper Text Transfer Protocol
* It is a protocol that allows interaction between two internet connected computers.

## what is web server?
* It is a software that accepts the web request.

## HTTP verbs 
In browser perspective,
* GET - retrieve data
* POST - create data
* DELETE - delete data with the help of id
* PUT - update data

In server perspective,
* GET - send data
* POST - receives data

## REST principles
* It is about how the response is sent for the web request.
* it does not give html data.
* instead, it provides data in the form of JSON.
* it is stateless - it knows only the current request not the previous request.

## Flask in python
* It is an API which understand the request and decides what to do with the request.

## Steps to create Flask application
import Flask
```
from flask import Flask
```
create object for Flask
```
app = Flask(__name__)
```
create endpoint and define method
```
@app.route('/')
def printHelloWorld():
   return "Hello World!!!"
```
run the app
```
app.run()
```



