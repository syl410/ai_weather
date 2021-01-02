var express = require("express");
var router  = express.Router();
var geocoder = require('geocoder');
var fs = require('fs');
var multer = require('multer');
var path = require('path');
// get image size
var sizeOf = require('image-size');
// a tool to change size of base64 image
var resizebase64 = require('resize-base64'); 
var im = require('imagemagick');
const sharp = require('sharp');


//About
router.get("/about", function(req, res){
   res.render("about"); 
});

// index
router.get("/", function(req, res){
	// weather information is read from a json file which is created by python script
	fs.readFile('./machine_learning_for_weather_forecast/new_forecast.json', (err, data) => {
		if (err) {
			console.error(err)
			return
		}
		var weather_info = JSON.parse(data);
		console.log(weather_info);
		res.render("index", {weather_info : weather_info}); 
	});
});

    

module.exports = router;
