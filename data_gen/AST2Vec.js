// Sheneamer, A., Roy, S., & Kalita, J. (2018). 
// A detection framework for semantic code clones and obfuscated code.
// Expert Systems with Applications, 97, 405-420.


const { getAST } = require('./obfuscate')
const fs = require('fs')
const acorn = require('acorn')
const walk = require('acorn-walk');

const ITERATIONS = [
    'WhileStatement',
    'DoWhileStatement',
    'ForStatement',
    'ForInStatement',
    'ForOfStatement',
]

const EXPRESSIONS = [
    'Identifier',
    'Literal',
    'ThisExpression',
    'ArrayExpression',
    'ObjectExpression',
    'FunctionExpression',
    'UnaryExpression',
    'UpdateExpression',
    'BinaryExpression',
    'LogicalExpression',
    'MemberExpression',
    'ConditionalExpression',
    'CallExpression',
    'NewExpression',
    'SequenceExpression',
    'ArrowFunctionExpression',
    'YieldExpression',
    'TemplateLiteral',
    'TaggedTemplateExpression',
    'ClassExpression',
    'MetaProperty',
    'AwaitExpression',
    'ChainExpression',
    'ImportExpression',
    'ParenthesizedExpression',
]

// https://ars.els-cdn.com/content/image/1-s2.0-S0957417417308631-mmc1.pdf
const AST_FEATURE_TEMPLATE = [
    'AssignmentExpression', // assignments
    'IfStatement', // selections
    ...ITERATIONS,
    'ReturnStatement', // return
    'SwitchCase', // switch
    'SwitchStatement',
    'ContinueStatement',
    'BreakStatement',
    'TryStatement', // Try statements
    'CatchClause',
    'ThrowStatement',
    'VariableDeclaration', // Declarations
    'VariableDeclarator',
    'FunctionDeclaration',

    'Expression', // Expression
    ...EXPRESSIONS,
    'ClassDeclaration', // class declarations
    'Super',
    'Class',
    'ClassBody',
    'MethodDefinition',
    'ImportDeclaration',
    'ImportSpecifier',
    'ImportDefaultSpecifier',
    'ImportNamespaceSpecifier',
    'ExportNamedDeclaration',
    'ExportSpecifier',
    'AnonymousClassDeclaration',
    'ExportDefaultDeclaration',
    'ExportAllDeclaration',

    'Property',
    'SpreadElement',
    'Function', // function
    'AnonymousFunctionDeclaration',

    'BlockStatement', // others
    'EmptyStatement',
    'DebuggerStatement',
    'WithStatement',
    'LabeledStatement',
    'TemplateElement',
    'AssignmentProperty',
    'ObjectPattern',
    'ArrayPattern',
    'RestElement',
    'AssignmentPattern',
    'PropertyDefinition',
    'PrivateIdentifier',
    'StaticBlock',
]

/**
 * Get ast features
 * @param {acorn.Node} ast the ast node
 * @param {Object} features feature cache
 */
function get_ast_features(ast, features) {
    const visitors = {}
    for (let key of AST_FEATURE_TEMPLATE) {
        features[key] = 0
        visitors[key] = function(_, state) {
            state[key] += 1
        }
    }
    walk.simple(ast, visitors, undefined, features)
}

function ast2vec(features) {
    const vec = []
    vec.push(features['lines'])
    for (let key of AST_FEATURE_TEMPLATE) {
        vec.push(features[key])
    }
    return vec
}

if (typeof require !== 'undefined' && require.main === module) {
    const args = process.argv.slice(2)
    const inputPath = args[0]
    const outputPath = args[1]

    const js_file = fs.readFileSync(inputPath, 'utf8')
    const ast = getAST(js_file, true)

    let features = {}
    features['lines'] = js_file.split(acorn.lineBreak).length
    get_ast_features(ast, features)
    features['vec'] = ast2vec(features)

    fs.writeFileSync(outputPath, JSON.stringify(features))
}