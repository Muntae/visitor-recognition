var express = require('express'), 
    bodyParser = require('body-parser'),
    port = process.env.PORT || 3000;

var MongoClient = require("mongodb").MongoClient();

var multer  = require('multer'),
    admin = require("firebase-admin"),
    fs = require('fs')
    gcStorage = require('@google-cloud/storage');

var mongodb, time, created_at, YYYY, MM, DD, HH, mm, ss, imageUrl;
var lock = false;
var noti_term = 60; // 푸시 알람 간격(초)
var before_alarm_time;
var app = express();

var serverKey = 'AAAANg4sIEI:APA91bFWGOtJ4l4r6tPwm3UmIKzRExH8i9tdt8Tm3iWre9UwCe9CNhRBJoM9Ligg4LmOl3mb7x7ws6LIk_4PMVGhoKqzIbRbHIejZ5RVi0WPZlkRmqps4rd-1cwu43OBgs9hQfGVbxPf';

var dbConnect = MongoClient.connect("mongodb://admin:OEOLSGNYEQELBFJM@sl-us-south-1-portal.4.dblayer.com:18145,sl-us-south-1-portal.5.dblayer.com:18145/admin?ssl=true", function(err, db) {
  mongodb = db.db("examples");
});

app.set('view engine', 'ejs');
app.set('views', './views');

app.use(express.static(__dirname + '/public'));
app.use(bodyParser.urlencoded({
  extended: true
})); 

app.listen(port);
console.log("Server Start");

const gcs = gcStorage({
  keyFilename: './recog.json'
});

function leadingZeros(n, digits) {
  var zero = '';
  n = n.toString();

  if (n.length < digits) {
    for (i = 0; i < digits - n.length; i++)
      zero += '0';
  }
  return zero + n;
}

var storage = multer.diskStorage({
  destination: function(req, file, callback) {
    callback(null, './public/images')
  },
  filename: function(req, file, callback) {
    var now = new Date();
    tzOffset = 9;
    var tz = now.getTime() + (now.getTimezoneOffset() * 60000) + (tzOffset * 3600000);
    now.setTime(tz);
  
    YYYY = leadingZeros(now.getFullYear(), 4);
    MM = leadingZeros(now.getMonth() + 1, 2)
    DD = leadingZeros(now.getDate(), 2)
    HH = leadingZeros(now.getHours(), 2);
    mm = leadingZeros(now.getMinutes(), 2);
    ss = leadingZeros(now.getSeconds(), 2);
    time = now.getTime();
    created_at = YYYY + "-" + MM + "-" + DD + " " + HH + ":" + mm + ":" + ss;
    callback(null, time + '.jpg');
  }
});
var upload = multer({ storage: storage });

var serviceAccount = require("./visitor.json");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
  databaseURL: "https://visitor-recognition.firebaseio.com"
});

var registrationToken = "c3Oa8wiX7nI:APA91bGAGIGElxI7iHggTRSDNgknRn806cYieVp0Djz7fBhUpT4TiVreCayoECqaEvK5MOJs004zjEx96AEvcxOSsaGYc2K_VdFPCV3dDDUg1fqkivnkBb9U_1IbMscbSb_KAUIuCU-7";

var payload = {
  "notification" : {
    "body" : "사람이 감지 되었습니다.",
    "title" : "[알림]",
    "sound" : "default"
  }
};

app.get("/", function(req, res) {
});

app.post("/create", upload.any(), function(req, res) {
  const bucketName = 'endurance';
  const filename = time + '.jpg';
  const filepath = './public/images/' + filename;
  imageUrl = 'endurance/' + filename;
  gcs
  .bucket(bucketName)
  .upload(filepath)
  .then(() => {
    console.log(`${filepath} uploaded to ${bucketName}.`);
    fs.unlinkSync(filepath);
    gcs
    .bucket(bucketName)
    .file(filename)
    .makePublic()
    .then(() => {
      // console.log(`gs://${bucketName}/${filename} is now public.`);
    })
    .catch((err) => {
      console.error('ERROR:', err);
    });
  })
  .catch((err) => {
    console.error('ERROR:', err);
  });

  mongodb.collection("dataTest").insertOne({
    name: time,
    img: req.files,
    created_at: created_at,
    YYYY: YYYY,
    MM: MM,
    DD: DD,
    HH: HH,
    mm: mm,
    ss: ss,
    imageUrl: imageUrl
  }, function(err, r) {
  });
  alarm_time = YYYY.toString() + MM.toString() + DD.toString() + HH.toString() + mm.toString() + ss.toString();
  if (lock == true) {
    console.log("Not Send notification");
    console.log(alarm_time - before_alarm_time);
    // 첫 푸시 이후 몇분동안 푸시를 다시 보내지 않을지
    if (alarm_time - before_alarm_time > noti_term) { // 1분이 지나면 다시 보냄
      lock = false;
      console.log("Open the lock");
    }
  }
  if (lock == false) {
    admin.messaging().sendToDevice(registrationToken, payload)
    .then(function(res) {
      console.log("Successfully sent message:", res.results);
    })
    .catch(function(error) {
      console.log("Error sending message:", error);
    });
    before_alarm_time = alarm_time;
    lock = true;
    console.log("Send notification");
  }  
  res.end();
});

app.get("/read", function(req, res) {
  mongodb.collection("dataTest").find().toArray(function(err, info) {
    if (err) {
      res.status(500).send(err);
    } else {
      var first_data;
      var data_set = [];
      var j = 0, k = 1;
      for (var i = 0; i < info.length; i++) {
        if (i == 0) {
          first_data = info[i];
          data_set[j] = new Array();
          data_set[j][0] = first_data;
        }
        if (data_set[j][0].name <= info[i].name && info[i].name <= data_set[j][0].name + 1000 * 60 * 1) {
          data_set[j][k] = info[i];
          // console.log("데이터 찍을 때 data_set: " + data_set[j][k].created_at + "// info[i]: " + info[i]);
          k++;
        } else {
          first_data = info[i];
          data_set[++j] = new Array();
          data_set[j][0] = first_data;
          k = 1;
          // console.log("else: " + data_set[j]);
        }
      }
      res.render("view", {data_set: data_set});
    }
  });
});

app.post("/read", function(req, res) {
  mongodb.collection("dataTest").find().toArray(function(err, info) {
    if (err) {
      res.status(500).send(err);
    } else {
      var index = req.body.condition;
      var first_data;
      var data_set = [];
      var j = 0, k = 1;
      for (var i = 0; i < info.length; i++) {
        if (i == 0) {
          first_data = info[i];
          data_set[j] = new Array();
          data_set[j][0] = first_data;
        }
        if (data_set[j][0].name <= info[i].name && info[i].name <= data_set[j][0].name + 1000 * 60 * 1) {
          data_set[j][k] = info[i];
          // console.log("데이터 찍을 때 data_set: " + data_set[j][k].created_at + "// info[i]: " + info[i]);
          k++;
        } else {
          first_data = info[i];
          data_set[++j] = new Array();
          data_set[j][0] = first_data;
          k = 1;
          // console.log("else: " + data_set[j]);
        }
      }
      res.render("read", {data_set: data_set, index: index});
    }
  });
});

require("cf-deployment-tracker-client").track();