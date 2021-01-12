// this is a server file and it will start a server
// It will 1, setup server, 2, connect DB, 3, routes

// it is true if the server is run locally.
var isLocal = false;
// require takes the name of a package as a string argument and returns a package
var express     = require("express");

// spawn child process asynchronously.
const {spawn} = require('child_process');

// Calls the express function "express()" and 
// puts new Express application inside the app var (to start a new Express application). 
var app         = express();

// body-parser extract the entire body portion of an incoming request stream and exposes it on req.body
var bodyParser  = require("body-parser");

var cookieParser = require("cookie-parser");
var LocalStrategy = require("passport-local");

var methodOverride = require("method-override");

// get current time with specific timezone
var moment = require('moment-timezone');

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
// the app will search content in public directory
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


function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function wait() {
	var hasRunPyToday = false; // true if python has been run today
	var isReboot = "True"; // a string passed to python. isReboot == "True" if this is called first time the app is deployed
	while(true) {
		var time_now = moment().tz("America/Chicago").format(); // 2020-03-28T00:17:38-05:00
		var HH_MM = time_now.substr(11, 5); // 00:17 (hours:minutes)
		await sleep(20000); // 1 is 1ms, 1000 is 1s
		// console.log(HH_MM);
		if (isReboot == "True" || HH_MM == "23:01") {
			if (!hasRunPyToday) {
				console.log("Start running python");

				// e.g.: var process = spawn('python',["./hello.py", req.query.firstname, req.query.lastname]);
				const pyProcess = spawn('python', ['./machine_learning_for_weather_forecast/web_collect_process_predict.py', isReboot]);
				// const pyProcess = spawn('python', ['./machine_learning_for_weather_forecast/hi.py']);
				// Takes stdout data from script which executed 
				// with arguments
				pyProcess.stdout.on('data', function(output) { 
				    console.log(output.toString()); // buffer to string
				});
				pyProcess.stderr.on('data', function(output) {
    				console.error(output.toString());
				});

				if (isReboot != "True") {
					hasRunPyToday = true;
				}
				isReboot = "false";
			}
		} else {
			hasRunPyToday = false;
		}
	}
}


// port of cloud9 environment: process.env.PORT
// local is http://localhost:8080/
var port = isLocal ? 8080 : process.env.PORT; 
app.listen(port, process.env.IP, function(){
   console.log("AI weather Server Has Started!");
   wait();
});


// var time = new Date();
// console.log(time.getHours() + ":" + time.getMinutes());
