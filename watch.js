const chokidar = require('chokidar')
const spawn = require('child_process').spawn


const compileAndRun = 
    () => spawn('make -j 4 && ./bin/a.out',{ stdio: 'inherit',shell:  true } )

function clearConsole() {
    console.log('\033c')
}

let process = compileAndRun()

chokidar.watch(__dirname, { ignored: /^(node_modules)|(bin)|(.git)/ }).on('change', (event, path) => {
    process.kill()
    clearConsole()
    process = compileAndRun()
})
