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
   res.render("index"); 
});

    

module.exports = router;