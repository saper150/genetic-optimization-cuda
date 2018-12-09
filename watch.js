const chokidar = require('chokidar')
const spawn = require('child_process').spawn


const compileAndRun = 
    () => spawn('make -j 4 && ./bin/a.out',{ stdio: 'inherit',shell:  true } )

let process = compileAndRun()

chokidar.watch(__dirname, { ignored: /^(node_modules)|(bin)/ }).on('change', (event, path) => {
    process.kill()
    console.log('\033c')
    process = compileAndRun()
})
