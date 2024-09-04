const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { Parser } = require("acorn");
const bigInt = require('acorn-bigint');
const JavaScriptObfuscator = require('javascript-obfuscator')
const { readTokens, readAST, readASTLoose, getAST } = require("./index");

/**
 * Search all js files in given directory and copy them to a given folder
 * @param {string} filePath the path to the directory that contains js files
 * @param {string} prefix the prefix of the file
 * @param {string} destination the destination folder path
 */
async function copyFiles(filePath, prefix, destination) {
    const dir = await fs.promises.opendir(filePath);
    let index = 0;
    for await (const dirent of dir) {
        const name = dirent.name;
        if (name.endsWith('js') && dirent.isFile()) {
            const newName = prefix + '-' + index + '.js';
            fs.copyFile(path.join(filePath, name), path.join(destination, newName), (err) => {
                if (err) console.error(`${path.join(filePath, name)} has error: ${err}`);
            });
        } else if (dirent.isDirectory()) {
            await copyFiles(path.join(filePath, name), prefix + '-' + index, destination);
        }
        index++;
    }
}

function readFile(filePath, config) {
    const content = fs.readFileSync(filePath, "utf8");
    const result = [];
    for (let token of Parser.extend(bigInt).tokenizer(content, { config })) {
        result.push(token.type.label);
    }
    return result;
}

 async function readFolder(folderPath, fileName, dest) {
    const filePath = path.join(folderPath, fileName);
    const fileStream = fs.createReadStream(filePath);

    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity
    });

    const script_config = {
        sourceType: 'script'
    };

    const module_config = {
        allowHashBang: true,
        sourceType: 'module',
        allowImportExportEverywhere: true
    };

    let script = 0, module = 0, failed = 0;
    for await (const line of rl) {
        if (line.startsWith('data/unknown')) {
            continue;
        }
        let tokens = [];
        try {
            tokens = readFile(path.join(folderPath, line), script_config);
            script++;
        } catch (e) {
            try {
                tokens = readFile(path.join(folderPath, line), module_config);
                module++;
            } catch (e) {
                if (e.name != 'SyntaxError' && e.code != 'ENOENT') {
                    console.log(line);
                    console.log(e);
                }
                failed++;
                continue;
            }
        }
        
        const length = tokens.length;
        fs.appendFileSync(dest, `"${line}",${length}\n`);
    }
    console.log(`script: ${script}, module: ${module}, failed: ${failed}`)
}

/**
 * Obfuscate the file
 * @param {string} fileContent the content of the file
 * @return {string} the obfuscated content
 */
function obfuscateA(fileContent) {
    return JavaScriptObfuscator.obfuscate(fileContent, {}).toString();
}

/**
 * Another obfuscation method
 * @param {string} code the content of the file
 * @return {string} the obfuscated content
 */
function obfuscateB(code) {
    return JavaScriptObfuscator.obfuscate(code, {
        compact: true,
        controlFlowFlattening: true,
        controlFlowFlatteningThreshold: 1,
        deadCodeInjection: true,
        deadCodeInjectionThreshold: 1,
        debugProtection: true,
        debugProtectionInterval: true,
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
        splitStringsChunkLength: 5,
        stringArray: true,
        stringArrayEncoding: 'rc4',
        stringArrayThreshold: 1,
        transformObjectKeys: true,
        unicodeEscapeSequence: false
    }).toString();
}

function obfuscateC(code) {
    return JavaScriptObfuscator.obfuscate(code, {
        compact: true,
        controlFlowFlattening: true,
        controlFlowFlatteningThreshold: 1,
        deadCodeInjection: true,
        deadCodeInjectionThreshold: 1,
        debugProtection: true,
        debugProtectionInterval: true,
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
        splitStringsChunkLength: 5,
        stringArray: true,
        stringArrayEncoding: 'rc4',
        stringArrayThreshold: 1,
        transformObjectKeys: true,
        unicodeEscapeSequence: false
    }).toString();
}

/**
 * Randomly pick a file name from the list
 * @param fileNames a list of file names
 * @param name the name to be excluded
 * @return {string} a randomly picked file name from the list
 */
function randomPickName(fileNames, name) {
    const index = Math.floor(Math.random() * fileNames.length);
    const newName = fileNames[index];
    if (newName === name) return randomPickName(fileNames, name);
    return newName;
}

/**
 * Mix files and obfuscate them
 * @param source the source folder
 * @param oneDest the destination folder for true pairs
 * @param zeroDest the destination folder for wrong pairs
 * @param silent true if does not print any error
 * @return {Promise<boolean>} true if no error happens
 */
async function mixAndObfuscate(source, oneDest, zeroDest, silent) {
    const dir = await fs.promises.opendir(source);
    const fileNames = [];
    for await (const dirent of dir) {
        const name = dirent.name;
        fileNames.push(name);
    }
    let anyError = false;
    for (const name of fileNames) {
        if (!fs.existsSync(path.join(oneDest, name))) {
            const content = fs.readFileSync(path.join(source, name), "utf8");
            const newName = randomPickName(fileNames, name);
            const newContent = fs.readFileSync(path.join(source, newName), "utf8");

            try {
                let fileContent = content + '\n(function () {\n' + newContent + '\n})();\n';
                const mixedContent = JavaScriptObfuscator.obfuscate(fileContent, {}).toString();
                fs.writeFileSync(path.join(oneDest, name), mixedContent);
            } catch (e) {
                anyError = true;
                if (!silent) console.error(`Error 1: ${name} - ${e}`)
            }

        }
        if (!fs.existsSync(path.join(zeroDest, name))) {
            const newName1 = randomPickName(fileNames, name);
            const newName2 = randomPickName(fileNames, name);
            const newContent1 = fs.readFileSync(path.join(source, newName1), "utf8");
            const newContent2 = fs.readFileSync(path.join(source, newName2), "utf8");

            try {
                let fileContents = newContent1 + '\n(function () {\n' + newContent2 + '\n})();\n';
                const mixedContents = JavaScriptObfuscator.obfuscate(fileContents, {}).toString();
                fs.writeFileSync(path.join(zeroDest, name), mixedContents);
            } catch (e) {
                anyError = true;
                if (!silent) console.error(`Error 0: ${name} - ${e}`)
            }
        }
    }
    return anyError;
}

async function obfuscateAll(source, oneDest, zeroDest, obsFunction, silent = false) {
    const dir = await fs.promises.opendir(source);
    const fileNames = [];
    for await (const dirent of dir) {
        const name = dirent.name;
        fileNames.push(name);
    }
    if (!fs.existsSync(oneDest)) {
        fs.mkdirSync(oneDest);
    }
    if (!fs.existsSync(zeroDest)) {
        fs.mkdirSync(zeroDest);
    }
    let result = 0;
    for (const name of fileNames) {
        if (!fs.existsSync(path.join(oneDest, name))) {
            const content = fs.readFileSync(path.join(source, name), "utf8");
            const newName = randomPickName(fileNames, name);
            const newContent = fs.readFileSync(path.join(source, newName), "utf8");

            try {
                let mixedContent = mixContents(content, newContent);
                const obsContent = obsFunction(mixedContent);
                fs.writeFileSync(path.join(oneDest, name), obsContent);
            } catch (e) {
                result++;
                if (!silent) {
                    console.error(`Error 1: ${name} - ${e}`)
                }
            }

        }
        if (!fs.existsSync(path.join(zeroDest, name))) {
            const newName1 = randomPickName(fileNames, name);
            const newName2 = randomPickName(fileNames, name);
            const newContent1 = fs.readFileSync(path.join(source, newName1), "utf8");
            const newContent2 = fs.readFileSync(path.join(source, newName2), "utf8");

            try {
                let fileContents = mixContents(newContent1, newContent2);
                const mixedContents = obsFunction(fileContents);
                fs.writeFileSync(path.join(zeroDest, name), mixedContents);
            } catch (e) {
                result++;
                if (!silent) {
                    console.error(`Error 0: ${name} - ${e}`)
                }
            }
        }
    }
    return result;
}

/**
 * Mix the seed at a random position into the target
 * @param {string} seed the content of the seed file
 * @param {string} target the content of the target file
 */
function mixContents(seed, target) {
    const ast = getAST(target, true);
    const size = ast.body.length;
    const position = Math.floor(Math.random() * Math.floor(size + 1));
    if (position === 0) {
        return seed + '\n' + target;
    } else if (position === size) {
        return target + '\n' + seed;
    } else {
        const element = ast.body[position];
        return target.substring(0, element.start) + '\n' + seed + '\n' + target.substring(element.start);
    }
}

/**
 * Tokenize the js files and write to given folder
 * @param {string} sourceDir the source directory
 * @param {string} destDir the dest directory
 * @return {Promise<void>}
 */
async function writeTokens(sourceDir, destDir) {
    const dir = await fs.promises.opendir(sourceDir);
    if (!fs.existsSync(destDir)) {
        fs.mkdirSync(destDir);
    }
    for await (const dirent of dir) {
        const name = dirent.name;
        if (fs.existsSync(path.join(destDir, name))) continue;
        let tokens;
        try {
            tokens = readTokens(path.join(sourceDir, name));
        } catch (e) {
            console.error(`${path.join(sourceDir, name)} cannot be tokenized: ${e}.`);
            continue;
        }

        const newContent = tokens.join(' ');
        fs.writeFile(path.join(destDir, name), newContent, (err) => {
            if (err) console.error(`${name} cannot be wrote to folder ${destDir}: ${err}.`);
        });
    }
}

/**
 * Parse the js files to ASTs and write to given folder
 * @param {string} sourceDir the source directory
 * @param {string} destDir the dest directory
 * @param {boolean} loose whether to be error tolerant
 * @return {Promise<void>}
 */
async function writeASTs(sourceDir, destDir, loose = false) {
    const dir = await fs.promises.opendir(sourceDir);
    for await (const dirent of dir) {
        const name = dirent.name;
        if (fs.existsSync(path.join(destDir, name))) continue;
        let ast;
        try {
            if (loose) {
                ast = readASTLoose(path.join(sourceDir, name));
            } else {
                ast = readAST(path.join(sourceDir, name));
            }
        } catch (e) {
            console.error(`${path.join(sourceDir, name)} cannot be parsed script to AST: ${e}.`);
        }

        const newContent = _stringify(ast);
        fs.writeFile(path.join(destDir, name), newContent, (err) => {
            if (err) console.error(`${name} cannot be wrote to folder ${destDir}: ${err}.`);
        });
    }
}

function _stringify(object) {
    return JSON.stringify(object, (key, value) =>
        typeof value === 'bigint' ?
        value.toString() :
        value // return everything else unchanged
    );
}

/**
 * Return a map from length in kilo to the number of files
 * @param {string} directory the directory to count file length
 * @returns {Promise<object<number, number>>} a map from length in kilo to the number of files
 */
async function countFileLength(directory) {
    const dir = await fs.promises.opendir(directory);
    const lengths = {};

    for await (const dirent of dir) {
        const name = dirent.name;
        const content = fs.readFileSync(path.join(directory, name), "utf8");
        const length = Math.ceil(content.length / 1000);
        if (lengths[length]) lengths[length]++;
        else lengths[length] = 1;
    }

    return lengths;
}

/**
 * Count files in given directory
 * @param {string} filePath the path to the folder
 * @returns {Promise<number>} number of files
 */
async function count(filePath) {
    let result = 0;
    const dir = await fs.promises.opendir(filePath);
    for await (const dirent of dir) {
        result++;
    }
    return result;
}

/**
 * Get the stats of the tokens
 * @param directory {string} the file directory
 * @returns {Promise<({}|number)[]>} a tuple of the stats
 */
async function readAllFileTokens(directory) {
    const dir = await fs.promises.opendir(directory);
    let max_length = 0;
    const lengths = {};
    const tokenMap = {};
    let numOfTokens = 0;

    for await (const dirent of dir) {
        const name = dirent.name;
        const tokens = readTokens(path.join(directory, name));
        const length = Math.ceil(tokens.length / 100);
        if (lengths[length]) lengths[length]++;
        else lengths[length] = 1;
        if (tokens.length > max_length) max_length = tokens.length;
        tokens.forEach((value) => {
            if (tokenMap[value]) tokenMap[value]++;
            else {
                tokenMap[value] = 1;
                numOfTokens++;
            }
        })
    }

    return [lengths, tokenMap, numOfTokens, max_length];
}

async function validateFiles(sourceDir, destDir) {
    const dir = await fs.promises.opendir(sourceDir);
    let size = 0;
    for await (const dirent of dir) {
        const name = dirent.name;
        size++;
        if (fs.existsSync(path.join(destDir, name))) continue;
        console.error(`${name} does not exist ${destDir}.`)
    }
    console.log(size);
}

module.exports = {
    mixAndObfuscate,
    writeTokens,
    copyFiles,
    count,
    obfuscateAll,
    readAllFileTokens,
    countFileLength
};

if (typeof require !== 'undefined' && require.main === module) {
    const args = process.argv.slice(2);
    const action = args[0];
    switch (action) {
        case 'obfuscate':
            let model = args[1];
            obfuscateAll('./origin', model + '1', model + '0', obfuscateA).then(result => {
                if (result) {
                    console.error("Error occurred " + result);
                } else {
                    console.log("Succeed")
                }
            })
            break;
        case 'tokens':
            model = args[1];
            writeTokens(model + '0', model + '0_tokens').catch(console.error)
            writeTokens(model + '1', model + '1_tokens').catch(console.error)
            break;
        case 'validate':
            model = args[1];
            validateFiles('./origin', model + '1').catch(console.error);
            validateFiles('./origin', model + '0').catch(console.error);
            validateFiles('./origin', model + '0_tokens').catch(console.error);
            validateFiles('./origin', model + '0_tokens').catch(console.error);
            break;
        case 'search':
            readFolder(args[1], args[2], args[3]).catch(console.error);
    }
}