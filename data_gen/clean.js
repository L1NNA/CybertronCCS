const escodegen = require('escodegen');
const { getAST } = require('./obfuscate')
const fs = require('fs')


if (typeof require !== 'undefined' && require.main === module) {
    const args = process.argv.slice(2)
    const inputPath = args[0]
    const outputPath = args[1]

    const js_file = fs.readFileSync(inputPath, 'utf8')
    const ast = getAST(js_file, true)

    const generatedCode = escodegen.generate(ast);

    fs.writeFileSync(outputPath, generatedCode)
}