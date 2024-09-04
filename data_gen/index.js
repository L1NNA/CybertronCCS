const fs = require('fs');
const ed = require('edit-distance');
const { Parser } = require("acorn");
const { LooseParser } = require("acorn-loose");
const bigInt = require('acorn-bigint');
const JavaScriptObfuscator = require('javascript-obfuscator')
const UglifyJS = require("uglify-js");


/**
 * Base configurations for JavaScript obfuscation
 */
base_config = {
  "compact": true,
  "config": "",
  "controlFlowFlattening": false,
  "controlFlowFlatteningThreshold": 0.75,
  "deadCodeInjection": false,
  "deadCodeInjectionThreshold": 0.4,
  "debugProtection": false,
  "debugProtectionInterval": false,
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
  "reservedNames": [],
  "reservedStrings": ['.*'],
  "rotateStringArray": false,
  "seed": 261767984,
  "selfDefending": false,
  "shuffleStringArray": false,
  "simplify": true,
  "sourceMap": false,
  "sourceMapBaseUrl": "",
  "sourceMapFileName": "",
  "sourceMapMode": "separate",
  "splitStrings": false,
  "splitStringsChunkLength": 9,
  "stringArray": false,
  "stringArrayEncoding": false,
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
 * Read tokens from the file
 * @param filePath the path to the file
 * @param isContent if filePath already is the content
 * @returns {[string]} a list of tokens
 */
function readTokens(filePath, isContent = false) {
    const content = isContent ? filePath : fs.readFileSync(filePath, "utf8");
    const result = [];
    for (let token of Parser.extend(bigInt).tokenizer(content, { allowHashBang: true })) {
        result.push(token.type.label);
    }
    return result;
}

function readOriginalTokens(filePath, isContent = false) {
    const content = isContent ? filePath : fs.readFileSync(filePath, "utf8");
    const result = [];
    const labels = ['name', '+/-', '</>/<=/>=', '!/~', '==/!=/===/!==', '++/--', '<</>>/>>>']
    for (let token of Parser.extend(bigInt).tokenizer(content, { allowHashBang: true })) {
        if (labels.indexOf(token.type.label) > -1) {
            result.push(token.value);
        } else if (token.type.label === 'string') {
            result.push(escape(token.value));
        } else {
            result.push(token.type.label);
        }
    }
    return result;
}

function removeRangeInfo(ast) {
    if (typeof ast === 'object') {
        if (Array.isArray(ast)) {
            const array = [];
            ast.forEach(el => {
                array.push(removeRangeInfo(el))
            });
            return array;
        } else {
            const object = {};
            for (const property in ast) {
                if (!ast.hasOwnProperty(property)) continue;
                if (property === 'start' || property === 'end') continue;
                object[property] = removeRangeInfo(ast[property]);
            }
            return object;
        }
    }
    return ast;
}

/**
 * Read the AST from the file
 * @param {string} filePath the path to the file
 * @returns {object} the ast tree for the js file
 */
function readAST(filePath) {
    const config = {
        allowHashBang: true,
        sourceType: 'module',
        allowImportExportEverywhere: true
    };
    const content = fs.readFileSync(filePath, "utf8");
    const ast = Parser.extend(bigInt).parse(content, config);
    return removeRangeInfo(ast);
}

/**
 * Read the loose AST from file
 * @param {string} filePath the path to the file
 * @returns {object} the ast tree for the js file
 */
function readASTLoose(filePath) {
    const config = {
        allowHashBang: true,
        sourceType: 'module',
        allowImportExportEverywhere: true
    };
    const content = fs.readFileSync(filePath, "utf8");
    const ast = LooseParser.extend(bigInt).parse(content, config);
    return removeRangeInfo(ast);
}

function _looseAST(filePath, isContent = false) {
    const content = isContent ? filePath : fs.readFileSync(filePath, "utf8");
    const ast = getAST(content, true);
    return removeRangeInfo(ast);

}

/**
 * Read AST strictly first. Then read it loosely if failed
 * @param content the file content
 * @param loose whether to read it loosely
 * @returns {Node} the AST node
 */
function getAST(content, loose) {
    const config = {
        allowHashBang: true,
        sourceType: 'module',
        allowImportExportEverywhere: true
    };
    let ast;
    try {
        ast = Parser.extend(bigInt).parse(content, config);
    } catch (e) {
        if (loose) {
            ast = LooseParser.extend(bigInt).parse(content, config);
        } else {
            throw e;
        }
    }
    return ast;
}


/**
 * Calculate the string distance between two strings
 * @param {string} stringA the first string
 * @param {string} stringB the second string
 * @returns {number} the Levenshtein Distance between two strings
 */
function stringDistance(stringA, stringB) {

    // Define cost functions.
    let insert, remove;
    insert = remove = function (node) {
        return 1;
    };
    const update = function (valueA, valueB) {
        return valueA !== valueB ? 1 : 0;
    };

    const lev = ed.levenshtein(stringA, stringB, insert, remove, update);
    return lev.distance
}

/**
 * Calculate the Levenshtein distance between two list of tokens
 * @param tokensA {[string]}
 * @param tokensB {[string]}
 * @returns {number} the Levenshtein Distance between two list of tokens
 */
function tokenDistance(tokensA, tokensB) {
    // populate zeros
    const lengthA = tokensA.length;
    const lengthB = tokensB.length;
    const result = []
    for (let i = 0; i < lengthA + 1; i++) {
        let temp = [];
        for (let j = 0; j < lengthB + 1; j++) {
            if (i === 0) temp.push(j);
            else if (j === 0) temp.push(i);
            else temp.push(0);
        }
        result.push(temp);
    }
    // levenshtein distance algorithm
    for (let j = 1; j < lengthB + 1; j++) {
        for (let i = 1; i < lengthA + 1; i++) {
            let substitutionCost = tokensA[i - 1] === tokensB[j - 1] ? 0 : 1;
            result[i][j] = Math.min(
                result[i - 1][j] + 1, // deletion
                result[i][j - 1] + 1, // insertion
                result[i - 1][j - 1] + substitutionCost); // substitution
        }
    }
    return result[lengthA][lengthB]
}

/**
 * Calculate the distance between two abstract syntax trees
 * @param {Array} astA the first array of ast
 * @param {Array} astB the second array of ast
 * @returns {number} the distance between two ASTs
 */
function ASTDistance(astA, astB) {
    // Define cost functions.
    let insert, remove;
    insert = remove = function (node) {
        return 1;
    };
    const update = function (valueA, valueB) {
        const keys = [];
        for (const property in valueA) {
            if (!valueA.hasOwnProperty(property)) continue;
            const value = valueA[property];
            if (typeof value === 'object') continue;
            keys.push(property);
            if (!valueB.hasOwnProperty(property)) return 1;
            if (value !== valueB[property]) return 1;
        }
        for (const property in valueB) {
            if (!valueB.hasOwnProperty(property)) continue;
            if (keys.indexOf(property) > -1) continue;
            if (typeof value === 'object') continue;
            if (!valueA.hasOwnProperty(property)) return 1;
        }
        return 0;
    };

    // Define two trees.
    const children = function (node) {
        const children = [];
        for (const property in node) {
            if (!node.hasOwnProperty(property)) continue;
            const value = node[property];
            if (typeof value === 'object') {
                if (Array.isArray(value)) {
                    value.forEach(el => {
                        children.push(el);
                    });
                } else {
                    children.push(value);
                }
            }

        }
        return children;
    };

    const ted = ed.ted(astA, astB, children, insert, remove, update);
    return ted.distance;
}

/**
 * Calculate the depth for an abstract syntax tree
 * @param {object} ast the abstract syntax tree
 * @returns {number} the depth of the abstract syntax tree
 */
function countAST(ast) {
    if (typeof ast === 'object') {
        let result = 1;
        if (Array.isArray(ast)) {
            ast.forEach(el => {
                result += countAST(el);
            });
        } else {
            for (const property in ast) {
                if (!ast.hasOwnProperty(property)) continue;
                result += countAST(ast[property]);
            }
        }
        return result;
    }
    return 0;
}

function _stringify(object) {
    return JSON.stringify(object, (key, value) =>
        typeof value === 'bigint' ?
            value.toString() :
            value // return everything else unchanged
    );
}

/**
 * Obfuscate the file
 * @param {string} code the content of the file
 * @return {string} the obfuscated content
 */
function obfuscatorLow(code) {
    return JavaScriptObfuscator.obfuscate(code, {
        compact: true,
        controlFlowFlattening: false,
        deadCodeInjection: false,
        debugProtection: false,
        debugProtectionInterval: false,
        disableConsoleOutput: true,
        identifierNamesGenerator: 'hexadecimal',
        log: false,
        numbersToExpressions: false,
        renameGlobals: false,
        rotateStringArray: true,
        selfDefending: true,
        shuffleStringArray: true,
        simplify: true,
        splitStrings: false,
        stringArray: true,

        stringArrayThreshold: 0.75,
        unicodeEscapeSequence: false
    }).toString();
}

/**
 * Another obfuscation method
 * @param {string} code the content of the file
 * @return {string} the obfuscated content
 */
function obfuscatorHigh(code) {
    return JavaScriptObfuscator.obfuscate(code, {
        compact: true,
        controlFlowFlattening: true,
        controlFlowFlatteningThreshold: 0.75,
        deadCodeInjection: true,
        deadCodeInjectionThreshold: 0.4,
        debugProtection: false,
        debugProtectionInterval: false,
        disableConsoleOutput: true,
        identifierNamesGenerator: 'hexadecimal',
        log: false,
        numbersToExpressions: true,
        renameGlobals: false,
        rotateStringArray: true,
        selfDefending: true,
        shuffleStringArray: true,
        simplify: true,
        splitStrings: true,
        splitStringsChunkLength: 10,
        stringArray: true,
        stringArrayEncoding: 'base64',
        stringArrayThreshold: 0.75,
        transformObjectKeys: true,
        unicodeEscapeSequence: false

    }).toString();
}

/**
 * Another obfuscation method
 * @param {string} code the content of the file
 * @return {string} the obfuscated content
 */
function obfuscatorMedium(code) {
    return JavaScriptObfuscator.obfuscate(code, {
        compact: true,
        controlFlowFlattening: true,
        controlFlowFlatteningThreshold: 0.75,
        deadCodeInjection: true,
        deadCodeInjectionThreshold: 0.4,
        debugProtection: false,
        debugProtectionInterval: false,
        disableConsoleOutput: true,
        identifierNamesGenerator: 'hexadecimal',
        log: false,
        numbersToExpressions: true,
        renameGlobals: false,
        rotateStringArray: true,
        selfDefending: true,
        shuffleStringArray: true,
        simplify: true,
        splitStrings: true,
        splitStringsChunkLength: 10,
        stringArray: true,
        stringArrayEncoding: 'base64',
        stringArrayThreshold: 0.75,
        transformObjectKeys: true,
        unicodeEscapeSequence: false
    }).toString();
}

/**
 * Another obfuscation method
 * @param {string} code the content of the file
 * @return {string} the obfuscated content
 */
function obfuscatorMini(code) {
    return UglifyJS.minify(code).code;
}

function mix_contents(seed, dest) {
    const dest_ast = getAST(dest, true);
    const dest_size = dest_ast.body.length;
    const position = Math.floor(Math.random() * Math.floor(dest_size + 1));
    if (position === 0) {
        return seed + '\n' + dest;
    } else if (position === dest_size) {
        return dest + '\n' + seed;
    } else {
        const element = dest_ast.body[position];
        return dest.substring(0, element.start) + '\n' + seed + '\n' + dest.substring(element.start);
    }
}

/**
 * Mix and obfuscate the seed at a random position into the target
 * @param obfuscator {function} the obfuscation method
 * @param {string} seedFile the file that holds the content of the seed file
 * @param {string} altFile the file that may or may not be the seed file
 * @param {string} destFile the file that holds the content of the target file
 * @param result {string} the result of the file
 * @param normalized {string} whether the token needs to be normalized
 * @param partial {boolean} whether to obfuscate alt only
 * @param mix {boolean} whether to mix alt into dest
 * @param to_ast {boolean} whether to output ast
 * @return {object}
 */
function obfuscate(obfuscator = obfuscatorLow, seedFile, altFile, destFile, result,
                   normalized, partial = false, mix = true, to_ast = false) {
    const seed = fs.readFileSync(seedFile, "utf8");
    const alt = fs.readFileSync(altFile, "utf8");
    const dest = fs.readFileSync(destFile, "utf8");

    const alt_seed = partial ? obfuscator(alt) : alt;
    let combination = mix ? mix_contents(alt_seed, dest) : alt_seed;
    const obfuscated_content = partial ? combination : obfuscator(combination);

    // const obfuscated_content = part1
    const tokenizer = normalized === '1' ? readTokens : readOriginalTokens;
    const obfuscated_tokens = tokenizer(obfuscated_content, true);
    const obfuscated_ast = _looseAST(obfuscated_content, true);
    const seed_tokens = tokenizer(seed, true);
    const seed_ast = _looseAST(seed, true);
    function ast_to_str(ast) {
        return JSON.stringify(ast.body[0], (key, value) =>
            typeof value === 'bigint' ? value.toString() : value
        );
    }
    output = {
        obfuscated_tokens: obfuscated_tokens.join(' '),
        seed_tokens: seed_tokens.join(' '),
        dist_string: [
            stringDistance(seed, obfuscated_content),
            seed.length, obfuscated_content.length
        ],
        dist_ast: [
            ASTDistance(seed_ast.body, obfuscated_ast.body),
            countAST(seed_ast.body), countAST(obfuscated_ast.body)
        ],
        dist_token: [
            tokenDistance(seed_tokens, obfuscated_tokens),
            seed_tokens.length, obfuscated_tokens.length
        ],
        result: result === '1' ? 1 : 0
    }
    if (to_ast) {
        output['obfuscated_ast'] = ast_to_str(obfuscated_ast)
        output['seed_ast'] = ast_to_str(seed_ast)
    }
    
    return output
}

function readOnePair(src_file, dest_file, normalized){
    const tokenizer = normalized === '1' ? readTokens : readOriginalTokens;
    const src = fs.readFileSync(src_file, "utf8");
    const dest = fs.readFileSync(dest_file, "utf8");
    const src_tkns = tokenizer(src, true);
    const dest_tkns = tokenizer(dest, true);
    const src_ast = _looseAST(src, true);
    const dest_ast = _looseAST(dest, true);
    return {
        seed_tokens: src_tkns.join(' '),
        obfuscated_tokens: dest_tkns.join(' '),
        dist_string: [
            stringDistance(src, dest),
            src.length, dest.length
        ],
        dist_ast: [
            ASTDistance(src_ast.body, dest_ast.body),
            countAST(src_ast.body), countAST(dest_ast.body)
        ],
        dist_token: [
            tokenDistance(src_tkns, dest_tkns),
            src_tkns.length, dest_tkns.length
        ],
        result: 0
    };
}


module.exports = {
    readTokens,
    readAST,
    readASTLoose,
    getAST,
}

if (typeof require !== 'undefined' && require.main === module) {
    const skip = 0;
    const model = process.argv[2 + skip];
    switch (model) {
        case 'token':
            console.log(readTokens(process.argv[3 + skip]).join(' '));
            break;
        case 'tokenOriginal':
            console.log(readOriginalTokens(process.argv[3 + skip]).join(' '));
            break;
        case 'token2':
            console.log(_stringify(readOnePair(process.argv[3 + skip], process.argv[4 + skip], process.argv[5 + skip])))
            break;
        case 'ast':
            console.log(_stringify(readAST(process.argv[3 + skip])));
            break;
        case 'ed_string':
            const content1 = fs.readFileSync(process.argv[3 + skip], "utf8");
            const content2 = fs.readFileSync(process.argv[4 + skip], "utf8");
            console.log(content1.length, content2.length, stringDistance(content1, content2))
            break;
        case 'ed_tokens':
            const tokens1 = readTokens(process.argv[3 + skip]);
            const tokens2 = readTokens(process.argv[4 + skip]);
            console.log(tokens1.length, tokens2.length, tokenDistance(tokens1, tokens2))
            break;
        case 'ed_ast':
            const ast1 = _looseAST(process.argv[3 + skip]);
            const ast2 = _looseAST(process.argv[4 + skip]);
            console.log(countAST(ast1.body), countAST(ast2.body), ASTDistance(ast1.body, ast2.body))
            break;
        case 'hi':
            console.log(_stringify(obfuscate(obfuscatorHigh, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip])));
            break;
        case 'low':
            console.log(_stringify(obfuscate(obfuscatorLow, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip])));
            break;
        case 'medium':
            console.log(_stringify(obfuscate(obfuscatorMedium, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip])));
            break;
        case 'minify':
            console.log(_stringify(obfuscate(obfuscatorMini, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip])));
            break;
        case 'string':
            _string_config = JSON.parse(JSON.stringify(base_config))
            _string_config['renameGlobals'] = true
            _string_config['splitStrings'] = true
            _string_config['reservedStrings'] = []
            obfuscation = code => JavaScriptObfuscator.obfuscate(code, _string_config).toString()
            console.log(_stringify(obfuscate(obfuscation, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip])));
            break;
        case 'control':
            _control_config = JSON.parse(JSON.stringify(base_config))
            _control_config['controlFlowFlattening'] = true
            obfuscation = code => JavaScriptObfuscator.obfuscate(code, _control_config).toString()
            console.log(_stringify(obfuscate(obfuscation, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip])));
            break;
        case 'control_dead':
            _control_config = JSON.parse(JSON.stringify(base_config))
            _control_config['controlFlowFlattening'] = true
            _control_config['deadCodeInjection'] = true
            obfuscation = code => JavaScriptObfuscator.obfuscate(code, _control_config).toString()
            console.log(_stringify(obfuscate(obfuscation, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip])));
            break;
        case 'string_dead':
            _control_config = JSON.parse(JSON.stringify(base_config))
            _control_config['renameGlobals'] = true
            _control_config['splitStrings'] = true
            _control_config['reservedStrings'] = []
            _control_config['deadCodeInjection'] = true
            obfuscation = code => JavaScriptObfuscator.obfuscate(code, _control_config).toString()
            console.log(_stringify(obfuscate(obfuscation, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip])));
            break;
        case 'object_dead':
            _control_config = JSON.parse(JSON.stringify(base_config))
            _control_config['transformObjectKeys'] = true
            _control_config['renameProperties'] = true
            _control_config['deadCodeInjection'] = true
            obfuscation = code => JavaScriptObfuscator.obfuscate(code, _control_config).toString()
            console.log(_stringify(obfuscate(obfuscation, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip])));
            break;
        case 'sco_dead':
            _control_config = JSON.parse(JSON.stringify(base_config))
            _control_config['transformObjectKeys'] = true
            _control_config['renameProperties'] = true
            _control_config['renameGlobals'] = true
            _control_config['splitStrings'] = true
            _control_config['reservedStrings'] = []
            _control_config['controlFlowFlattening'] = true
            _control_config['deadCodeInjection'] = true
            obfuscation = code => JavaScriptObfuscator.obfuscate(code, _control_config).toString()
            console.log(_stringify(obfuscate(obfuscation, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip])));
            break;
        case 'object':
            _control_config = JSON.parse(JSON.stringify(base_config))
            _control_config['transformObjectKeys'] = true
            _control_config['renameProperties'] = true
            obfuscation = code => JavaScriptObfuscator.obfuscate(code, _control_config).toString()
            console.log(_stringify(obfuscate(obfuscation, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip])));
            break;
        case 'sco':
            _control_config = JSON.parse(JSON.stringify(base_config))
            _control_config['transformObjectKeys'] = true
            _control_config['renameProperties'] = true
            _control_config['renameGlobals'] = true
            _control_config['splitStrings'] = true
            _control_config['reservedStrings'] = []
            _control_config['controlFlowFlattening'] = true
            obfuscation = code => JavaScriptObfuscator.obfuscate(code, _control_config).toString()
            console.log(_stringify(obfuscate(obfuscation, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip])));
            break;
        case 'p_control':
            _control_config = JSON.parse(JSON.stringify(base_config))
            _control_config['controlFlowFlattening'] = true
            obfuscation = code => JavaScriptObfuscator.obfuscate(code, _control_config).toString()
            console.log(_stringify(obfuscate(obfuscation, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip]), true));
            break;
        case 'w_control':
            _control_config = JSON.parse(JSON.stringify(base_config))
            _control_config['controlFlowFlattening'] = true
            obfuscation = code => JavaScriptObfuscator.obfuscate(code, _control_config).toString()
            console.log(_stringify(obfuscate(obfuscation, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip], true, false)));
            break;
        case 'w_control_ast':
            _control_config = JSON.parse(JSON.stringify(base_config))
            _control_config['controlFlowFlattening'] = true
            obfuscation = code => JavaScriptObfuscator.obfuscate(code, _control_config).toString()
            console.log(_stringify(obfuscate(obfuscation, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip], true, false, true)));
            break;
        case 'w_object':
            _control_config = JSON.parse(JSON.stringify(base_config))
            _control_config['transformObjectKeys'] = true
            _control_config['renameProperties'] = true
            obfuscation = code => JavaScriptObfuscator.obfuscate(code, _control_config).toString()
            console.log(_stringify(obfuscate(obfuscation, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip], true, false)));
            break;
        case 'w_string':
            _string_config = JSON.parse(JSON.stringify(base_config))
            _string_config['renameGlobals'] = true
            _string_config['splitStrings'] = true
            _string_config['reservedStrings'] = []
            obfuscation = code => JavaScriptObfuscator.obfuscate(code, _string_config).toString()
            console.log(_stringify(obfuscate(obfuscation, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip], true, false)));
            break;
        case 'w_sco':
            _control_config = JSON.parse(JSON.stringify(base_config))
            _control_config['transformObjectKeys'] = true
            _control_config['renameProperties'] = true
            _control_config['renameGlobals'] = true
            _control_config['splitStrings'] = true
            _control_config['reservedStrings'] = []
            _control_config['controlFlowFlattening'] = true
            obfuscation = code => JavaScriptObfuscator.obfuscate(code, _control_config).toString()
            console.log(_stringify(obfuscate(obfuscation, process.argv[3 + skip], process.argv[4 + skip],
                process.argv[5 + skip], process.argv[6 + skip], process.argv[7 + skip], true, false)));
            break;
    }
}