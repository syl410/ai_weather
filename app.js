// this is a server file and it will start a server
// It will 1, setup server, 2, connect DB, 3, routes

// require takes the name of a package as a string argument and returns a package
var express     = require("express");

// Calls the express function "express()" and 
// puts new Express application inside the app var (to start a new Express application). 
var app         = express();

// body-parser extract the entire body portion of an incoming request stream and exposes it on req.body
var bodyParser  = require("body-parser");

var cookieParser = require("cookie-parser");
var LocalStrategy = require("passport-local");

var methodOverride = require("method-override");
var fs = require('fs');
var multer = require('multer');
// configure dotenv
require('dotenv').load();

//requiring routes
var indexRoutes      = require("./routes/index");
    
// use is a method to configure the middleware used by the routes of the Express HTTP server object.
app.use(bodyParser.urlencoded({extended: true}));

// EJS simply stands for Embedded Javascript.
// It is a simple templating language/engine that lets its user generate HTML with plain javascript
app.set("view engine", "ejs");

// __dirname is directory path of currently executing js
app.use(express.static(__dirname + "/public"));

// https://dev.to/moz5691/method-override-for-put-and-delete-in-html-3fp2
app.use(methodOverride('_method'));
app.use(cookieParser('secret'));

//require moment
// Moment.js: use the native JavaScript Date object directly. 
app.locals.moment = require('moment');


app.use(function(req, res, next){
   res.locals.currentUser = req.user;
   next();
});

app.use("/", indexRoutes);

// port of cloud9 environment: process.env.PORT
app.listen(process.env.PORT, process.env.IP, function(){
   console.log("Austin Secondhand Server Has Started!");
});
