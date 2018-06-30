const http = require('http');
const fs = require('fs');
var myPythonScript = "SpellChecker.py";
var pythonExecutable = "python3";

const server = http.createServer((req, res) => {
    if (req.method === 'POST') {
        var body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        req.on('end', () => {
            fs.writeFileSync("input.txt", body);
            //res.writeHeader(200, {"Content-Type": "text/html"});




            var val = terminal();

            if(val>0)
                writing();

        });
    }

    else {
        res.end(`
            <!doctype html>
            <html>
            <head>
               <style type="text/css">
                    body {
                    background: black;
                    }
                    
                    p {
                        color: lightgray;
                        font-family: Arial;
                        font-size: 50px;
                    }
                    
                    form {
                        
                        height: 400px;
                        width: 100%;
                        
                                   
                    }
                    
                    div {
                       margin-top: 5%;
                       text-align: center;
                    }
                    
                    textarea {
                    border: 1px solid white;
                    border-radius: 5px;
                    width: 500px;
                    text-indent: 10px;
                    font-size: 15px;
                    padding-top: 10px;
                    }
                    
                    button {
                        cursor: pointer;
                        background: #34a853;
                        color: #f2f2f2;
                        border: 1px solid #34a853;
                        border-radius: 5px;
                        width: 100px;
                        padding: 20px;
                        margin-right: 5px;
                        font-size: 15px;
                        font-family: Arial;
                    }
                    
                    button:hover {
                        background: #f2f2f2;
                        color: #34a853;
                    }
                    
                    a {
                        text-decoration: none;
                        color : white;
                        width: 100px;
                        height: 300px;
                        padding : 18px 18px;
                        background:   #e74c3c ;
                        border: 1px solid   #e74c3c;
                        border-radius: 5px;
                        font-size: 15px;
                        font-family: Arial;
                    }
                    
                    a:hover {
                        background : #f2f2f2;
                        color: #e74c3c;
                    }
                    
                  
                </style>
            </head>
            <body>
            <div>
               <p>Spell Corrector<p>
                <form action="/" method="post">
                    <!--<textarea rows="15" name="fname"></textarea>-->
                    <textarea rows="15" name="fname"></textarea>
                    <br/>
                    <br/>
                    <button>Submit</button>
                    <a href="http://localhost:8000">Output</a>
                </form>
             </div>
            </body>
            </html>
        `);
    }
});
server.listen(3000);

function terminal() {


    // Function to convert an Uint8Array to a string
    var uint8arrayToString = function(data){
        return String.fromCharCode.apply(null, data);
    };

    const spawn = require('child_process').spawn;
    const scriptExecution = spawn(pythonExecutable, [myPythonScript]);

// Handle normal output
    scriptExecution.stdout.on('data', (data) => {
        console.log(uint8arrayToString(data));
    });

// Handle error output
    scriptExecution.stderr.on('data', (data) => {
        // As said before, convert the Uint8Array to a readable string.
        console.log(uint8arrayToString(data));
    });

    scriptExecution.on('exit', (code) => {
        console.log("Process quit with code : " + code);
    });
    return 1;
}

function writing() {
    http.createServer(function (req,res) {
        fs.readFile('output.txt', 'utf8', function (err, data) {
            res.write("OUTPUT: \n\n\n");
            res.end(data);
        });

    }).listen(8000);

}










