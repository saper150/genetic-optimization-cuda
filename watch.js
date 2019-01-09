const chokidar = require('chokidar')
const spawn = require('child_process').spawn
const fs = require('fs')

const ignored = new RegExp(
        ['.git'].concat(fs.readFileSync('.gitignore',{ encoding: 'utf8' }).split(/\r?\n/))
                .map(x=>`(${x})`).join('|')
    )

const compileAndRun = 
    () => spawn('make -j 4 && ./Optimization.out',{ stdio: 'inherit',shell:  true } )

function clearConsole() {
    console.log('\033c')
}

let process = compileAndRun()


chokidar.watch(__dirname, { ignored }).on('change', (event, path) => {
    process.kill()
    clearConsole()
    process = compileAndRun()
})
