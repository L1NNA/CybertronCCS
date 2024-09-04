const fs = require("fs");
const assert = require('assert');

const JavaScriptObfuscator = require('javascript-obfuscator')
const { Parser } = require("acorn");
const bigInt = require("acorn-bigint");
const { LooseParser } = require("acorn-loose");


/**
 * Base configurations for JavaScript obfuscation
 */
const BASE_CONFIG = {
    "compact": false,
    "config": "",
    "controlFlowFlattening": false,
    "controlFlowFlatteningThreshold": 0.75,
    "deadCodeInjection": false,
    "deadCodeInjectionThreshold": 0.4,
    "debugProtection": false,
    "debugProtectionInterval": 0,
    "disableConsoleOutput": false,
    "domainLock": [],
    "exclude": [],
    "forceTransformStrings": [],
    "identifierNamesGenerator": "hexadecimal",
    "identifiersPrefix": "",
    "identifiersDictionary": [],
    "ignoreRequireImports": false,
    "inputFileName": "",
    "log": false,
    "numbersToExpressions": false,
    "optionsPreset": "default",
    "renameGlobals": false,
    "renameProperties": false,
    "renamePropertiesMode": "safe",
    "reservedNames": ['^(?!_).*$'],
    "reservedStrings": ['.*'],
    "rotateStringArray": false,
    "seed": 261767984,
    "selfDefending": false,
    "shuffleStringArray": false,
    "simplify": false,
    "sourceMap": false,
    "sourceMapBaseUrl": "",
    "sourceMapFileName": "",
    "sourceMapMode": "separate",
    "splitStrings": false,
    "splitStringsChunkLength": 9,
    "stringArray": false,
    "stringArrayEncoding": ['none'],
    "stringArrayIndexesType": [
        "hexadecimal-number"
    ],
    "stringArrayIndexShift": false,
    "stringArrayWrappersChainedCalls": false,
    "stringArrayWrappersCount": 0,
    "stringArrayWrappersParametersMaxCount": 2,
    "stringArrayWrappersType": "variable",
    "stringArrayThreshold": 0,
    "target": "browser",
    "transformObjectKeys": false,
    "unicodeEscapeSequence": false,
    "splitStringsChunkLengthEnabled": true,
    "rotateStringArrayEnabled": false,
    "shuffleStringArrayEnabled": false,
    "stringArrayThresholdEnabled": false,
    "stringArrayEncodingEnabled": false,
    "domainLockEnabled": true
}

/**
 * Code Transformations
 */
const Transformations = {
    Renaming: 0,
    ObjectKeyTransformation: 1,
    StringArray: 2,
    StringSplitting: 3,
    ValueSubstituation: 4,
    ControlFlowFlattening: 5,
    DeadCodeInjection: 6
}

/**
 * Code Mixing Strategies
 */
const MixingStrategies = {
    M1: 1,
    M2: 2,
    M3: 3
}

/**
 * Get the arguments for given code transformation instructions
 * @param {Array[string]} transformations an array of code transformation instructions
 * @returns {object} the configurations for JS obfuscator
 */
function copyArguments(transformations) {
    const baseConfig = JSON.parse(JSON.stringify(BASE_CONFIG))

    for (let transformation of transformations) {
        switch (transformation) {
            case Transformations.Renaming:
                baseConfig.reservedNames = []
                baseConfig.renameGlobals = true
                break
            case Transformations.ObjectKeyTransformation:
                baseConfig.transformObjectKeys = true
                break
            case Transformations.StringArray:
                baseConfig.reservedStrings = []
                baseConfig.stringArray = true
                break
            case Transformations.StringSplitting:
                baseConfig.reservedStrings = []
                baseConfig.splitStrings = true
                break
            case Transformations.ValueSubstituation:
                baseConfig.reservedStrings = []
                break
            case Transformations.ControlFlowFlattening:
                baseConfig.controlFlowFlattening = true
                break
            case Transformations.DeadCodeInjection:
                baseConfig.deadCodeInjection = true
                break
        }
    }

    return baseConfig
}

/**
 * Read AST strictly first. Then read it loosely if failed
 * @param content the file content
 * @param loose whether to read it loosely
 * @returns {Node} the AST node
 */
function getAST(content, loose) {
    let config = {
        allowHashBang: true,
        sourceType: 'script',
        allowImportExportEverywhere: true,
        ecmaVersion: 2020
    };

    let excep;
    
    try {
        return Parser.extend(bigInt).parse(content, config);
    } catch (e) {
        excep = e
    }

    try {
        config.sourceType = 'module'
        return Parser.extend(bigInt).parse(content, config);
    } catch (e) {
        excep = e
    }

    
    if (loose) {
        config.sourceType = 'script'
        return LooseParser.extend(bigInt).parse(content, config);
    } else {
        throw excep;
    }
}

/**
 * Inject the seed script into the base script at a random location
 * @param {string} seed the seed script
 * @param {string} base the base script
 * @returns {string} the mixed script (target) 
 */
function code_injection(seed, base) {
    const base_ast = getAST(base, true);
    const base_size = base_ast.body.length;
    const position = Math.floor(Math.random() * Math.floor(base_size + 1));
    if (position === 0) {
        return seed + '\n' + base;
    } else if (position === base_size) {
        return base + '\n' + seed;
    } else {
        const element = base_ast.body[position];
        return base.substring(0, element.start) + '\n' + seed + '\n' + base.substring(element.start);
    }
}

/**
 * Mix the seed script into the base script at a random location
 * @param {string} seed the seed script
 * @param {string} base the base script
 * @returns {string} the mixed script (target) 
 */
function code_mixing(seed, base) {
    const base_ast = getAST(base, true);
    const seed_ast = getAST(seed, true)
    const base_size = base_ast.body.length;
    const seed_size = seed_ast.body.length

    const randomLocations = new Array(seed_size);
    for (let i = 0; i < seed_size; i++) {
        randomLocations[i] = Math.floor(Math.random() * (base_size + 1));
    }
    randomLocations.sort((a, b) => a - b);

    let result = ''
    let prev = 0
    for (let i = 0; i < seed_size; i++) {
        let position = randomLocations[i]
        if (position > prev) {
            let start = base_ast.body[prev].start
            let end = base_ast.body[position - 1].end
            result += base.substring(start, end)
            result += '\n'
        }
        
        let start = seed_ast.body[i].start
        let end = seed_ast.body[i].end
        result += seed.substring(start, end)
        if (position < base_size) {
            result += '\n'
        }
        prev = position
    }

    if (prev < base_size) {
        let start = base_ast.body[prev].start
        let end = base_ast.body[base_size - 1].end
        result += base.substring(start, end)
    }

    return result
}

module.exports = {getAST}

if (typeof require !== 'undefined' && require.main === module) {
    const args = process.argv.slice(2)
    const exp = args[0] // type of experiment
    const mixing_strategy = args[1] // type of mixing
    const seed_path = args[2] // seed file path
    const base_path = args[3] // base file path (can be undefined)
    const target_path = args[4] // output file path

    let transformations = []
    if (exp === 'Exp_all' || exp == 'Exp_var') {
        transformations.push(Transformations.Renaming)
        transformations.push(Transformations.ObjectKeyTransformation)
    }
    if (exp === 'Exp_all' || exp === 'Exp_val') {
        transformations.push(Transformations.StringArray)
        transformations.push(Transformations.StringSplitting)
        transformations.push(Transformations.ValueSubstituation)
    }
    if (exp === 'Exp_all' || exp === 'Exp_ast') {
        transformations.push(Transformations.ControlFlowFlattening)
    }

    assert.ok(transformations.length, 'Unknown experiment <' + exp + '>')
    assert.ok(mixing_strategy in MixingStrategies,
        'Unknown mixing strategy <' + mixing_strategy + '>')
    mixing = MixingStrategies[mixing_strategy]

    const seed = fs.readFileSync(seed_path, "utf8")
    let target = seed

    if (mixing > MixingStrategies.M1) {
        base = fs.readFileSync(base_path, "utf8")
        target = code_injection(seed, base)
    }
    if (mixing === MixingStrategies.M3) {
        transformations.push(Transformations.DeadCodeInjection)
    }

    let baseConfig = copyArguments(transformations)
    
    target = JavaScriptObfuscator.obfuscate(target, baseConfig).toString()
    fs.writeFileSync(target_path, target);
}

