var expect = require('chai').expect;
var nock = require('nock');
var fs = require('fs');
var ObjectStorage = require('../lib');

describe('ObjectStorage', function(){
  var os;

  beforeEach(function(){
      os = new ObjectStorage('userid', 'password', 'projectid', 'container');
  });

  afterEach(function(){
    nock.cleanAll();
  });

  describe('constructor', function(){
    it('should create an object of type ObjectStorage', function(){
      var os1 = new ObjectStorage('d', 'e', 'f', 'g');
      expect(os1.userId).to.equal('d');
      expect(os1.password).to.equal('e');
      expect(os1.projectId).to.equal('f');
      expect(os1.container).to.equal('g');
      expect(os1.endpoint).to.equal('https://dal.objectstorage.open.softlayer.com/v1/AUTH_f');
    });
  });

  describe('#createContainer', function(){
    it('should create a container', function(done){
      nock('https://dal.objectstorage.open.softlayer.com').put('/v1/AUTH_projectid/container').reply(201);
      os.createContainer().then(function(data){
        expect(data).to.be.null;
        done();
      });
    });
  });

    describe('#listContainerFiles', function(){
    it('should return a list of files in a container', function(done){
      nock('https://dal.objectstorage.open.softlayer.com').get('/v1/AUTH_projectid/container').reply(200, [3, 5, 7]);
      os.listContainerFiles().then(function(data){
        expect(data).to.have.length(3);
        done();
      });
    });
  });

    describe('#setContainerPublicReadable', function(){
    it('should make a container read only to the public', function(done){
      nock('https://dal.objectstorage.open.softlayer.com').post('/v1/AUTH_projectid/container').reply(204);
      os.setContainerPublicReadable().then(function(data){
        expect(data).to.be.null;
        done();
      });
    });
  });

  describe('#uploadFileToContainer', function(){
    it('should upload a file to the container', function(done){
      var buffer = fs.readFileSync('./test/fixtures/pup.jpg');
      nock('https://dal.objectstorage.open.softlayer.com').put('/v1/AUTH_projectid/container/pup.jpg').reply(201);
      os.uploadFileToContainer('pup.jpg', 'image/jpeg', buffer, 32796).then(function(data){
        expect(data).to.equal('https://dal.objectstorage.open.softlayer.com/v1/AUTH_projectid/container/pup.jpg');
        done();
      });
    });
  });
});
