const { Deobfuscator } = require("deobfuscator");
const fs = require('fs/promises')

async function deobfuscate(inputPath, outputPath) {
    
    
    const content = await fs.readFile(inputPath, "utf8")
    const deobs = new Deobfuscator()

    let result
    try {
        result = await deobs.deobfuscateSource(content)
    } catch (_){
        result = await deobs.deobfuscateSource(content, {loose:true})
    }
    await fs.writeFile(outputPath, result)
}

if (typeof require !== 'undefined' && require.main === module) {
    const args = process.argv.slice(2)
    const inputPath = args[0]
    const outputPath = args[1]

    deobfuscate(inputPath, outputPath)
}



