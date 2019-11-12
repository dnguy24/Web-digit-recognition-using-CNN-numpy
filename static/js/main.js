var mycanvas = document.getElementById("mycanvas")
var ctx = mycanvas.getContext("2d")
var rect = mycanvas.getBoundingClientRect()
var prevX = 0,
    prevY = 0,
    currX = 0,
    currY = 0
var data = []
var clear = false
function change(){
    clear = true
    var jsonData = JSON.stringify(ctx.getImageData(0, 0, mycanvas.clientWidth, mycanvas.clientHeight))
    $.ajax({
        type: "POST",
        contentType: "application/json;charset=utf-8",
        url: "data",
        traditional: "true",
        data: jsonData,
        dataType: "json",
        success: function (data) {
            console.log("yes!")
            document.getElementById("demo").innerHTML = data
        },
        error: function (data) {
            console.log("no:(")
            document.getElementById("result").innerHTML = "the digit is not recognizable, try again"
        }
    });
}

var draw = false
mycanvas.addEventListener("mousedown", function (e) {
    if(clear){
        ctx.clearRect(0, 0, mycanvas.clientWidth, mycanvas.clientHeight);
        clear = false;
    }
    prevX = currX;
    prevY = currY;
    currX = parseInt(e.clientX - rect.left);
    currY = parseInt(e.clientY - rect.top);
    data = [[currX, currY]]
    draw = true;
});

mycanvas.addEventListener("mouseup", function () {
    draw = false;
    // console.log(data)
})

mycanvas.addEventListener("mouseleave", function () {
    if(draw==true) {
        draw = false;
        clear = true;
        var jsonData = JSON.stringify(ctx.getImageData(0, 0, mycanvas.clientWidth, mycanvas.clientHeight))
    $.ajax({
        type: "POST",
        contentType: "application/json;charset=utf-8",
        url: "data",
        traditional: "true",
        data: jsonData,
        dataType: "json",
        success: function (data) {
            console.log("yes!")
            document.getElementById("result").innerHTML = ""
            document.getElementById("demo").innerHTML = data
        },
        error: function (data) {
            console.log("no:(")
            document.getElementById("result").innerHTML = "the digit is not recognizable, try again"
        }
    });
    }
})
mycanvas.addEventListener("mousemove", function(e){
    if(draw) {
        prevX = currX;
        prevY = currY;
        currX = parseInt(e.clientX - rect.left);
        currY = parseInt(e.clientY - rect.top);
        data.push([currX, currY])
        ctx.beginPath();
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.moveTo(prevX, prevY);
        ctx.lineTo(currX, currY);
        ctx.strokeStyle = "black";
        ctx.stroke();
        ctx.closePath();
    }
})

function download(content, fileName, contentType) {
    var a = document.createElement("a");
    var file = new Blob([content], {type: contentType});
    a.href = URL.createObjectURL(file);
    a.download = fileName;
    a.click();
}