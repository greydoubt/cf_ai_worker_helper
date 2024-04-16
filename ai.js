/* eslint-disable */
// ../worker-constellation-entry/src/ai/tensor.ts
var TypedArrayProto = Object.getPrototypeOf(Uint8Array);
function isArray(value) {
  return Array.isArray(value) || value instanceof TypedArrayProto;
}

// base structures
function ensureShape(shape, value) {
  if (shape.length === 0 && !isArray(value)) {
    return;
  }
  const count = shape.reduce((acc, v) => {
    if (!Number.isInteger(v)) {
      throw new Error(
        `expected shape to be array-like of integers but found non-integer element "${v}"`
      );
    }
    return acc * v;
  }, 1);
  if (count != arrLength(value)) {
    throw new Error(
      `invalid shape: expected ${count} elements for shape ${shape} but value array has length ${value.length}`
    );
  }
}




function ensureType(type, value) {
  if (isArray(value)) {
    value.forEach(v => ensureType(type, v));
    return;
  }
  switch (type) {
    case 'bool' /* Bool */: {
      if (typeof value === 'boolean') {
        return;
      }
      break;
    }
    case 'float16' /* Float16 */:
    case 'float32' /* Float32 */: {
      if (typeof value === 'number') {
        return;
      }
      break;
    }
    case 'int8' /* Int8 */:
    case 'uint8' /* Uint8 */:
    case 'int16' /* Int16 */:
    case 'uint16' /* Uint16 */:
    case 'int32' /* Int32 */:
    case 'uint32' /* Uint32 */: {
      if (Number.isInteger(value)) {
        return;
      }
      break;
    }
    case 'int64' /* Int64 */:
    case 'uint64' /* Uint64 */: {
      if (typeof value === 'bigint') {
        return;
      }
      break;
    }
    case 'str' /* String */: {
      if (typeof value === 'string') {
        return;
      }
      break;
    }
  }
  throw new Error(`unexpected type "${type}" with value "${value}".`);
}





// check array lengths
function arrLength(obj) {
  return obj instanceof TypedArrayProto
    ? obj.length
    : obj
        .flat(Infinity)
        .reduce(
          (acc, cur) => acc + (cur instanceof TypedArrayProto ? cur.length : 1),
          0
        );
}



function serializeType(type, value) {
  if (isArray(value)) {
    return [...value].map(v => serializeType(type, v));
  }
  switch (type) {
    case 'str' /* String */:
    case 'bool' /* Bool */:
    case 'float16' /* Float16 */:
    case 'float32' /* Float32 */:
    case 'int8' /* Int8 */:
    case 'uint8' /* Uint8 */:
    case 'int16' /* Int16 */:
    case 'uint16' /* Uint16 */:
    case 'uint32' /* Uint32 */:
    case 'int32' /* Int32 */: {
      return value;
    }
    case 'int64' /* Int64 */:
    case 'uint64' /* Uint64 */: {
      return value.toString();
    }
  }
  throw new Error(`unexpected type "${type}" with value "${value}".`);
}




function deserializeType(type, value) {
  if (isArray(value)) {
    return value.map(v => deserializeType(type, v));
  }
  switch (type) {
    case 'str' /* String */:
    case 'bool' /* Bool */:
    case 'float16' /* Float16 */:
    case 'float32' /* Float32 */:
    case 'int8' /* Int8 */:
    case 'uint8' /* Uint8 */:
    case 'int16' /* Int16 */:
    case 'uint16' /* Uint16 */:
    case 'uint32' /* Uint32 */:
    case 'int32' /* Int32 */: {
      return value;
    }
    case 'int64' /* Int64 */:
    case 'uint64' /* Uint64 */: {
      return BigInt(value);
    }
  }
  throw new Error(`unexpected type "${type}" with value "${value}".`);
}
var Tensor = class _Tensor {
  type;
  value;
  name;
  shape;
  constructor(type, value, opts = {}) {
    this.type = type;
    this.value = value;
    ensureType(type, this.value);
    if (opts.shape === void 0) {
      if (isArray(this.value)) {
        this.shape = [arrLength(value)];
      } else {
        this.shape = [];
      }
    } else {
      this.shape = opts.shape;
    }
    ensureShape(this.shape, this.value);
    this.name = opts.name || null;
  }
  static fromJSON(obj) {
    const { type, shape, value, b64Value, name } = obj;
    const opts = { shape, name };
    if (b64Value !== void 0) {
      const value2 = b64ToArray(b64Value, type)[0];
      return new _Tensor(type, value2, opts);
    } else {
      return new _Tensor(type, deserializeType(type, value), opts);
    }
  }
  toJSON() {
    return {
      type: this.type,
      shape: this.shape,
      name: this.name,
      value: serializeType(this.type, this.value)
    };
  }
};
function b64ToArray(base64, type) {
  const byteString = atob(base64);
  const bytes = new Uint8Array(byteString.length);
  for (let i = 0; i < byteString.length; i++) {
    bytes[i] = byteString.charCodeAt(i);
  }
  const arrBuffer = new DataView(bytes.buffer).buffer;
  switch (type) {
    case 'float32':
      return new Float32Array(arrBuffer);
    case 'float64':
      return new Float64Array(arrBuffer);
    case 'int32':
      return new Int32Array(arrBuffer);
    case 'int64':
      return new BigInt64Array(arrBuffer);
    default:
      throw Error(`invalid data type for base64 input: ${type}`);
  }
}

// ../worker-constellation-entry/src/ai/templates.ts
var tgTemplates = {
  // ex: https://huggingface.co/TheBloke/deepseek-coder-6.7B-base-AWQ
  bare: {
    system: {
      flag: 2 /* ABSORB_ROLE */
    },
    user: {
      flag: 3 /* APPEND_LAST_SYSTEM */
    },
    assistant: {
      pre: ' ',
      post: ' '
    }
  },
  //
  sqlcoder: {
    system: {
      flag: 2 /* ABSORB_ROLE */
    },
    user: {
      flag: 2 /* ABSORB_ROLE */
    },
    assistant: {
      flag: 2 /* ABSORB_ROLE */
    },
    global: {
      template:
        '### Task\nGenerate a SQL query to answer [QUESTION]{user}[/QUESTION]\n\n### Database Schema\nThe query will run on a database with the following schema:\n{system}\n\n### Answer\nGiven the database schema, here is the SQL query that [QUESTION]{user}[/QUESTION]\n[SQL]'
    }
  },
  // ex: https://huggingface.co/TheBloke/LlamaGuard-7B-AWQ
  inst: {
    system: {
      flag: 2 /* ABSORB_ROLE */
    },
    user: {
      pre: '[INST] ',
      post: ' [/INST]',
      flag: 3 /* APPEND_LAST_SYSTEM */
    },
    assistant: {
      pre: ' ',
      post: ' '
    }
  },
  // https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L340-L361
  // https://replicate.com/blog/how-to-prompt-llama
  // https://huggingface.co/TheBloke/Llama-2-13B-chat-AWQ#prompt-template-llama-2-chat
  llama2: {
    system: {
      pre: '[INST] <<SYS>>\n',
      post: '\n<</SYS>>\n\n'
    },
    user: {
      pre: '<s>[INST] ',
      post: ' [/INST]',
      flag: 1 /* CARRY_SYSTEM_INST */
    },
    assistant: {
      pre: ' ',
      post: '</s>'
    }
  },
  // https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-AWQ
  deepseek: {
    system: {
      post: '\n'
    },
    user: {
      pre: '### Instruction:\n',
      post: '\n'
    },
    assistant: {
      pre: '### Response:\n',
      post: '\n'
    },
    global: {
      post: '### Response:\n'
    }
  },
  // https://huggingface.co/TheBloke/Falcon-7B-Instruct-GPTQ
  falcon: {
    system: {
      post: '\n'
    },
    user: {
      pre: 'User: ',
      post: '\n'
    },
    assistant: {
      pre: 'Assistant: ',
      post: '\n'
    },
    global: {
      post: 'Assistant: \n'
    }
  },
  // https://huggingface.co/TheBloke/openchat_3.5-AWQ#prompt-template-openchat
  openchat: {
    system: {
      flag: 2 /* ABSORB_ROLE */
    },
    user: {
      pre: 'GPT4 User: ',
      post: '<|end_of_turn|>',
      flag: 3 /* APPEND_LAST_SYSTEM */
    },
    assistant: {
      pre: 'GPT4 Assistant: ',
      post: '<|end_of_turn|>'
    },
    global: {
      post: 'GPT4 Assistant:'
    }
  },
  // https://huggingface.co/openchat/openchat#conversation-template
  'openchat-alt': {
    system: {
      flag: 2 /* ABSORB_ROLE */
    },
    user: {
      pre: '<s>Human: ',
      post: '<|end_of_turn|>',
      flag: 3 /* APPEND_LAST_SYSTEM */
    },
    assistant: {
      pre: 'Assistant: ',
      post: '<|end_of_turn|>'
    },
    global: {
      post: 'Assistant: '
    }
  },
  // https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
  tinyllama: {
    system: {
      pre: '<|system|>\n',
      post: '</s>\n'
    },
    user: {
      pre: '<|user|>\n',
      post: '</s>\n'
    },
    assistant: {
      pre: '<|assistant|>\n',
      post: '</s>\n'
    },
    global: {
      post: '<|assistant|>\n'
    }
  },
  // https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-AWQ#prompt-template-chatml
  // https://huggingface.co/TheBloke/Orca-2-13B-AWQ#prompt-template-chatml
  chatml: {
    system: {
      pre: '<|im_start|>system\n',
      post: '<|im_end|>\n'
    },
    user: {
      pre: '<|im_start|>user\n',
      post: '<|im_end|>\n'
    },
    assistant: {
      pre: '<|im_start|>assistant\n',
      post: '<|im_end|>\n'
    },
    global: {
      post: '<|im_start|>assistant\n'
    }
  },
  // https://huggingface.co/TheBloke/neural-chat-7B-v3-1-AWQ#prompt-template-orca-hashes
  'orca-hashes': {
    system: {
      pre: '### System:\n',
      post: '\n\n'
    },
    user: {
      pre: '### User:\n',
      post: '\n\n'
    },
    assistant: {
      pre: '### Assistant:\n',
      post: '\n\n'
    },
    global: {
      post: '### Assistant:\n\n'
    }
  },
  // https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-AWQ#prompt-template-codellama
  'codellama-instruct': {
    system: {
      pre: '[INST] ',
      post: '\n'
    },
    user: {
      pre: '[INST] ',
      post: ' [/INST]\n',
      flag: 1 /* CARRY_SYSTEM_INST */
    },
    assistant: {
      post: '\n'
    }
  },
  // https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-AWQ#prompt-template-mistral
  'mistral-instruct': {
    system: {
      pre: '<s>[INST] ',
      post: ' '
    },
    user: {
      pre: '[INST] ',
      post: ' [/INST]',
      flag: 1 /* CARRY_SYSTEM_INST */
    },
    assistant: {
      pre: ' ',
      post: '</s>'
    }
  },
  // https://huggingface.co/TheBloke/zephyr-7B-beta-AWQ#prompt-template-zephyr
  // https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha
  zephyr: {
    system: {
      pre: '<s><|system|>\n',
      post: '</s>\n'
    },
    user: {
      pre: '<|user|>\n',
      post: '</s>\n'
    },
    assistant: {
      pre: '<|assistant|>\n',
      post: '</s>\n'
    },
    global: {
      post: '<|assistant|>\n'
    }
  }
};
var generateTgTemplate = (messages, template) => {
  let prompt = '';
  const state = {
    lastSystem: false,
    systemCount: 0,
    lastUser: false,
    userCount: 0,
    lastAssistant: false,
    assistantCount: 0
  };
  for (const message of messages) {
    switch (message.role) {
      case 'system':
        state.systemCount++;
        state.lastSystem = message.content;
        prompt += applyRole(template, message.role, message.content, state);
        break;
      case 'user':
        state.userCount++;
        state.lastUser = message.content;
        prompt += applyRole(template, message.role, message.content, state);
        break;
      case 'assistant':
        state.assistantCount++;
        state.lastAssistant = message.content;
        prompt += applyRole(template, message.role, message.content, state);
        break;
    }
  }
  prompt = applyRole(template, 'global', prompt, state);
  return prompt;
};
var applyTag = (template, role, type, state) => {
  if (
    type == 'pre' &&
    tgTemplates[template][role].flag == 1 /* CARRY_SYSTEM_INST */ &&
    state.systemCount == 1 &&
    state.userCount == 1
  ) {
    return '';
  }
  return tgTemplates[template][role][type] || '';
};
var applyRole = (template, role, content, state) => {
  if (tgTemplates[template] && tgTemplates[template][role]) {
    if (tgTemplates[template][role].flag == 2 /* ABSORB_ROLE */) return '';
    if (
      tgTemplates[template][role].flag == 3 /* APPEND_LAST_SYSTEM */ &&
      state.lastSystem &&
      state.userCount == 1
    ) {
      content = `${state.lastSystem}${
        [':', '.', '!', '?'].indexOf(state.lastSystem.slice(-1)) == -1
          ? ':'
          : ''
      } ${content}`;
    }
    if (tgTemplates[template][role].template) {
      return tgTemplates[template][role].template
        .replaceAll('{user}', state.lastUser)
        .replaceAll('{system}', state.lastSystem)
        .replaceAll('{assistant}', state.lastAssistant);
    }
    return (
      applyTag(template, role, 'pre', state) +
      (content || '') +
      applyTag(template, role, 'post', state)
    );
  }
  return content || '';
};

// ../worker-constellation-entry/src/ai/tasks/text-generation.ts
var AiTextGeneration = class {
  modelSettings;
  inputs;
  preProcessedInputs;
  postProcessedOutputs;
  tensors;
  // run ./scripts/gen-validators.ts if you change the schema
  schema = {
    input: {
      type: 'object',
      oneOf: [
        {
          properties: {
            prompt: {
              type: 'string',
              maxLength: 4096
            },
            raw: {
              type: 'boolean',
              default: false
            },
            stream: {
              type: 'boolean',
              default: false
            },
            max_tokens: {
              type: 'integer',
              default: 256
            }
          },
          required: ['prompt']
        },
        {
          properties: {
            messages: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  role: {
                    type: 'string'
                  },
                  content: {
                    type: 'string',
                    maxLength: 4096
                  }
                },
                required: ['role', 'content']
              }
            },
            stream: {
              type: 'boolean',
              default: false
            },
            max_tokens: {
              type: 'integer',
              default: 256
            }
          },
          required: ['messages']
        }
      ]
    },
    output: {
      oneOf: [
        {
          type: 'object',
          contentType: 'application/json',
          properties: {
            response: {
              type: 'string'
            }
          }
        },
        {
          type: 'string',
          contentType: 'text/event-stream',
          format: 'binary'
        }
      ]
    }
  };
  constructor(inputs, modelSettings2) {
    this.inputs = inputs;
    this.modelSettings = modelSettings2 || {
      experimental: true,
      inputsDefaultsStream: {
        max_tokens: 512
      },
      inputsDefaults: {
        max_tokens: 256
      },
      preProcessingArgs: {
        promptTemplate: 'inst',
        defaultContext: ''
      }
    };
  }
  preProcessing() {
    if (this.inputs.stream && this.modelSettings.inputsDefaultsStream) {
      this.inputs = {
        ...this.modelSettings.inputsDefaultsStream,
        ...this.inputs
      };
    } else if (this.modelSettings.inputsDefaults) {
      this.inputs = { ...this.modelSettings.inputsDefaults, ...this.inputs };
    }
    let prompt = '';
    if (this.inputs.messages === void 0) {
      if (this.inputs.raw == true) {
        prompt = this.inputs.prompt;
      } else {
        prompt = generateTgTemplate(
          [
            {
              role: 'system',
              content: this.modelSettings.preProcessingArgs.defaultContext
            },
            { role: 'user', content: this.inputs.prompt }
          ],
          this.modelSettings.preProcessingArgs.promptTemplate
        );
      }
    } else {
      prompt = generateTgTemplate(
        this.inputs.messages,
        this.modelSettings.preProcessingArgs.promptTemplate
      );
    }
    this.preProcessedInputs = {
      prompt,
      max_tokens: this.inputs.max_tokens,
      stream: this.inputs.stream ? true : false
    };
  }
  generateTensors(preProcessedInputs) {
    if (this.modelSettings.generateTensorsFunc) {
      return this.modelSettings.generateTensorsFunc(preProcessedInputs);
    } else {
      return [
        new Tensor('str' /* String */, [preProcessedInputs.prompt], {
          shape: [1],
          name: 'INPUT_0'
        }),
        new Tensor('uint32' /* Uint32 */, [preProcessedInputs.max_tokens], {
          // sequence length
          shape: [1],
          name: 'INPUT_1'
        })
      ];
    }
  }
  postProcessing(response) {
    if (this.modelSettings.postProcessingFunc) {
      this.postProcessedOutputs = {
        response: this.modelSettings.postProcessingFunc(
          response,
          this.preProcessedInputs
        )
      };
    } else {
      this.postProcessedOutputs = { response: response.name.value[0] };
    }
  }
  postProcessingStream(response, inclen) {
    if (this.modelSettings.postProcessingFuncStream) {
      return {
        response: this.modelSettings.postProcessingFuncStream(
          response,
          this.preProcessedInputs,
          inclen
        )
      };
    } else {
      return { response: response.name.value[0] };
    }
  }
};

// ../worker-constellation-entry/src/ai/tasks/text-classification.ts
var AiTextClassification = class {
  modelSettings;
  inputs;
  preProcessedInputs;
  postProcessedOutputs;
  tensors;
  // run ./scripts/gen-validators.ts if you change the schema
  schema = {
    input: {
      type: 'object',
      properties: {
        text: {
          type: 'string'
        }
      },
      required: ['text']
    },
    output: {
      type: 'array',
      contentType: 'application/json',
      items: {
        type: 'object',
        properties: {
          score: {
            type: 'number'
          },
          label: {
            type: 'string'
          }
        }
      }
    }
  };
  constructor(inputs, modelSettings2) {
    this.inputs = inputs;
    this.modelSettings = modelSettings2;
  }
  preProcessing() {
    this.preProcessedInputs = this.inputs;
  }
  generateTensors(preProcessedInputs) {
    if (this.modelSettings.generateTensorsFunc) {
      return this.modelSettings.generateTensorsFunc(preProcessedInputs);
    } else {
      return [
        new Tensor('str' /* String */, [preProcessedInputs.text], {
          shape: [1],
          name: 'input_text'
        })
      ];
    }
  }
  postProcessing(response) {
    this.postProcessedOutputs = [
      {
        label: 'NEGATIVE',
        score: response.logits.value[0][0]
      },
      {
        label: 'POSITIVE',
        score: response.logits.value[0][1]
      }
    ];
  }
};

// ../worker-constellation-entry/src/ai/tasks/text-embeddings.ts
var AiTextEmbeddings = class {
  modelSettings;
  inputs;
  preProcessedInputs;
  postProcessedOutputs;
  tensors;
  // run ./scripts/gen-validators.ts if you change the schema
  schema = {
    input: {
      type: 'object',
      properties: {
        text: {
          oneOf: [
            { type: 'string' },
            {
              type: 'array',
              items: {
                type: 'string'
              },
              maxItems: 100
            }
          ]
        }
      },
      required: ['text']
    },
    output: {
      type: 'object',
      contentType: 'application/json',
      properties: {
        shape: {
          type: 'array',
          items: {
            type: 'number'
          }
        },
        data: {
          type: 'array',
          items: {
            type: 'array',
            items: {
              type: 'number'
            }
          }
        }
      }
    }
  };
  constructor(inputs, modelSettings2) {
    this.inputs = inputs;
    this.modelSettings = modelSettings2;
  }
  preProcessing() {
    this.preProcessedInputs = this.inputs;
  }
  generateTensors(preProcessedInputs) {
    if (this.modelSettings.generateTensorsFunc) {
      return this.modelSettings.generateTensorsFunc(preProcessedInputs);
    } else {
      return [
        new Tensor(
          'str' /* String */,
          Array.isArray(preProcessedInputs.text)
            ? preProcessedInputs.text
            : [preProcessedInputs.text],
          {
            shape: [
              Array.isArray(preProcessedInputs.text)
                ? preProcessedInputs.text.length
                : [preProcessedInputs.text].length
            ],
            name: 'input_text'
          }
        )
      ];
    }
  }
  postProcessing(response) {
    if (this.modelSettings.postProcessingFunc) {
      this.postProcessedOutputs = this.modelSettings.postProcessingFunc(
        response,
        this.preProcessedInputs
      );
    } else {
      this.postProcessedOutputs = {
        shape: response.embeddings.shape,
        data: response.embeddings.value
      };
    }
  }
};


// cursor



function deserializeType(type, value) {
  if (isArray(value)) {
    return value.map(v => deserializeType(type, v));
  }
  switch (type) {
    case 'str' /* String */:
    case 'bool' /* Bool */:
    case 'float16' /* Float16 */:
    case 'float32' /* Float32 */:
    case 'int8' /* Int8 */:
    case 'uint8' /* Uint8 */:
    case 'int16' /* Int16 */:
    case 'uint16' /* Uint16 */:
    case 'uint32' /* Uint32 */:
    case 'int32' /* Int32 */: {
      return value;
    }
    case 'int64' /* Int64 */:
    case 'uint64' /* Uint64 */: {
      return BigInt(value);
    }
  }
  throw new Error(`unexpected type "${type}" with value "${value}".`);
}
var Tensor = class _Tensor {
  type;
  value;
  name;
  shape;
  constructor(type, value, opts = {}) {
    this.type = type;
    this.value = value;
    ensureType(type, this.value);
    if (opts.shape === void 0) {
      if (isArray(this.value)) {
        this.shape = [arrLength(value)];
      } else {
        this.shape = [];
      }
    } else {
      this.shape = opts.shape;
    }
    ensureShape(this.shape, this.value);
    this.name = opts.name || null;
  }
  static fromJSON(obj) {
    const { type, shape, value, b64Value, name } = obj;
    const opts = { shape, name };
    if (b64Value !== void 0) {
      const value2 = b64ToArray(b64Value, type)[0];
      return new _Tensor(type, value2, opts);
    } else {
      return new _Tensor(type, deserializeType(type, value), opts);
    }
  }
  toJSON() {
    return {
      type: this.type,
      shape: this.shape,
      name: this.name,
      value: serializeType(this.type, this.value)
    };
  }
};
function b64ToArray(base64, type) {
  const byteString = atob(base64);
  const bytes = new Uint8Array(byteString.length);
  for (let i = 0; i < byteString.length; i++) {
    bytes[i] = byteString.charCodeAt(i);
  }
  const arrBuffer = new DataView(bytes.buffer).buffer;
  switch (type) {
    case 'float32':
      return new Float32Array(arrBuffer);
    case 'float64':
      return new Float64Array(arrBuffer);
    case 'int32':
      return new Int32Array(arrBuffer);
    case 'int64':
      return new BigInt64Array(arrBuffer);
    default:
      throw Error(`invalid data type for base64 input: ${type}`);
  }
}

// ../worker-constellation-entry/src/ai/templates.ts
var tgTemplates = {
  // ex: https://huggingface.co/TheBloke/deepseek-coder-6.7B-base-AWQ
  bare: {
    system: {
      flag: 2 /* ABSORB_ROLE */
    },
    user: {
      flag: 3 /* APPEND_LAST_SYSTEM */
    },
    assistant: {
      pre: ' ',
      post: ' '
    }
  },
  //
  sqlcoder: {
    system: {
      flag: 2 /* ABSORB_ROLE */
    },
    user: {
      flag: 2 /* ABSORB_ROLE */
    },
    assistant: {
      flag: 2 /* ABSORB_ROLE */
    },
    global: {
      template:
        '### Task\nGenerate a SQL query to answer [QUESTION]{user}[/QUESTION]\n\n### Database Schema\nThe query will run on a database with the following schema:\n{system}\n\n### Answer\nGiven the database schema, here is the SQL query that [QUESTION]{user}[/QUESTION]\n[SQL]'
    }
  },
  // ex: https://huggingface.co/TheBloke/LlamaGuard-7B-AWQ
  inst: {
    system: {
      flag: 2 /* ABSORB_ROLE */
    },
    user: {
      pre: '[INST] ',
      post: ' [/INST]',
      flag: 3 /* APPEND_LAST_SYSTEM */
    },
    assistant: {
      pre: ' ',
      post: ' '
    }
  },
  // https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L340-L361
  // https://replicate.com/blog/how-to-prompt-llama
  // https://huggingface.co/TheBloke/Llama-2-13B-chat-AWQ#prompt-template-llama-2-chat
  llama2: {
    system: {
      pre: '[INST] <<SYS>>\n',
      post: '\n<</SYS>>\n\n'
    },
    user: {
      pre: '<s>[INST] ',
      post: ' [/INST]',
      flag: 1 /* CARRY_SYSTEM_INST */
    },
    assistant: {
      pre: ' ',
      post: '</s>'
    }
  },
  // https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-AWQ
  deepseek: {
    system: {
      post: '\n'
    },
    user: {
      pre: '### Instruction:\n',
      post: '\n'
    },
    assistant: {
      pre: '### Response:\n',
      post: '\n'
    },
    global: {
      post: '### Response:\n'
    }
  },
  // https://huggingface.co/TheBloke/Falcon-7B-Instruct-GPTQ
  falcon: {
    system: {
      post: '\n'
    },
    user: {
      pre: 'User: ',
      post: '\n'
    },
    assistant: {
      pre: 'Assistant: ',
      post: '\n'
    },
    global: {
      post: 'Assistant: \n'
    }
  },
  // https://huggingface.co/TheBloke/openchat_3.5-AWQ#prompt-template-openchat
  openchat: {
    system: {
      flag: 2 /* ABSORB_ROLE */
    },
    user: {
      pre: 'GPT4 User: ',
      post: '<|end_of_turn|>',
      flag: 3 /* APPEND_LAST_SYSTEM */
    },
    assistant: {
      pre: 'GPT4 Assistant: ',
      post: '<|end_of_turn|>'
    },
    global: {
      post: 'GPT4 Assistant:'
    }
  },
  // https://huggingface.co/openchat/openchat#conversation-template
  'openchat-alt': {
    system: {
      flag: 2 /* ABSORB_ROLE */
    },
    user: {
      pre: '<s>Human: ',
      post: '<|end_of_turn|>',
      flag: 3 /* APPEND_LAST_SYSTEM */
    },
    assistant: {
      pre: 'Assistant: ',
      post: '<|end_of_turn|>'
    },
    global: {
      post: 'Assistant: '
    }
  },
  // https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
  tinyllama: {
    system: {
      pre: '<|system|>\n',
      post: '</s>\n'
    },
    user: {
      pre: '<|user|>\n',
      post: '</s>\n'
    },
    assistant: {
      pre: '<|assistant|>\n',
      post: '</s>\n'
    },
    global: {
      post: '<|assistant|>\n'
    }
  },
  // https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-AWQ#prompt-template-chatml
  // https://huggingface.co/TheBloke/Orca-2-13B-AWQ#prompt-template-chatml
  chatml: {
    system: {
      pre: '<|im_start|>system\n',
      post: '<|im_end|>\n'
    },
    user: {
      pre: '<|im_start|>user\n',
      post: '<|im_end|>\n'
    },
    assistant: {
      pre: '<|im_start|>assistant\n',
      post: '<|im_end|>\n'
    },
    global: {
      post: '<|im_start|>assistant\n'
    }
  },
  // https://huggingface.co/TheBloke/neural-chat-7B-v3-1-AWQ#prompt-template-orca-hashes
  'orca-hashes': {
    system: {
      pre: '### System:\n',
      post: '\n\n'
    },
    user: {
      pre: '### User:\n',
      post: '\n\n'
    },
    assistant: {
      pre: '### Assistant:\n',
      post: '\n\n'
    },
    global: {
      post: '### Assistant:\n\n'
    }
  },
  // https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-AWQ#prompt-template-codellama
  'codellama-instruct': {
    system: {
      pre: '[INST] ',
      post: '\n'
    },
    user: {
      pre: '[INST] ',
      post: ' [/INST]\n',
      flag: 1 /* CARRY_SYSTEM_INST */
    },
    assistant: {
      post: '\n'
    }
  },
  // https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-AWQ#prompt-template-mistral
  'mistral-instruct': {
    system: {
      pre: '<s>[INST] ',
      post: ' '
    },
    user: {
      pre: '[INST] ',
      post: ' [/INST]',
      flag: 1 /* CARRY_SYSTEM_INST */
    },
    assistant: {
      pre: ' ',
      post: '</s>'
    }
  },
  // https://huggingface.co/TheBloke/zephyr-7B-beta-AWQ#prompt-template-zephyr
  // https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha
  zephyr: {
    system: {
      pre: '<s><|system|>\n',
      post: '</s>\n'
    },
    user: {
      pre: '<|user|>\n',
      post: '</s>\n'
    },
    assistant: {
      pre: '<|assistant|>\n',
      post: '</s>\n'
    },
    global: {
      post: '<|assistant|>\n'
    }
  }
};
var generateTgTemplate = (messages, template) => {
  let prompt = '';
  const state = {
    lastSystem: false,
    systemCount: 0,
    lastUser: false,
    userCount: 0,
    lastAssistant: false,
    assistantCount: 0
  };
  for (const message of messages) {
    switch (message.role) {
      case 'system':
        state.systemCount++;
        state.lastSystem = message.content;
        prompt += applyRole(template, message.role, message.content, state);
        break;
      case 'user':
        state.userCount++;
        state.lastUser = message.content;
        prompt += applyRole(template, message.role, message.content, state);
        break;
      case 'assistant':
        state.assistantCount++;
        state.lastAssistant = message.content;
        prompt += applyRole(template, message.role, message.content, state);
        break;
    }
  }
  prompt = applyRole(template, 'global', prompt, state);
  return prompt;
};
var applyTag = (template, role, type, state) => {
  if (
    type == 'pre' &&
    tgTemplates[template][role].flag == 1 /* CARRY_SYSTEM_INST */ &&
    state.systemCount == 1 &&
    state.userCount == 1
  ) {
    return '';
  }
  return tgTemplates[template][role][type] || '';
};
var applyRole = (template, role, content, state) => {
  if (tgTemplates[template] && tgTemplates[template][role]) {
    if (tgTemplates[template][role].flag == 2 /* ABSORB_ROLE */) return '';
    if (
      tgTemplates[template][role].flag == 3 /* APPEND_LAST_SYSTEM */ &&
      state.lastSystem &&
      state.userCount == 1
    ) {
      content = `${state.lastSystem}${
        [':', '.', '!', '?'].indexOf(state.lastSystem.slice(-1)) == -1
          ? ':'
          : ''
      } ${content}`;
    }
    if (tgTemplates[template][role].template) {
      return tgTemplates[template][role].template
        .replaceAll('{user}', state.lastUser)
        .replaceAll('{system}', state.lastSystem)
        .replaceAll('{assistant}', state.lastAssistant);
    }
    return (
      applyTag(template, role, 'pre', state) +
      (content || '') +
      applyTag(template, role, 'post', state)
    );
  }
  return content || '';
};

// ../worker-constellation-entry/src/ai/tasks/text-generation.ts
var AiTextGeneration = class {
  modelSettings;
  inputs;
  preProcessedInputs;
  postProcessedOutputs;
  tensors;
  // run ./scripts/gen-validators.ts if you change the schema
  schema = {
    input: {
      type: 'object',
      oneOf: [
        {
          properties: {
            prompt: {
              type: 'string',
              maxLength: 4096
            },
            raw: {
              type: 'boolean',
              default: false
            },
            stream: {
              type: 'boolean',
              default: false
            },
            max_tokens: {
              type: 'integer',
              default: 256
            }
          },
          required: ['prompt']
        },
        {
          properties: {
            messages: {
              type: 'array',
              items: {
                type: 'object',
                properties: {
                  role: {
                    type: 'string'
                  },
                  content: {
                    type: 'string',
                    maxLength: 4096
                  }
                },
                required: ['role', 'content']
              }
            },
            stream: {
              type: 'boolean',
              default: false
            },
            max_tokens: {
              type: 'integer',
              default: 256
            }
          },
          required: ['messages']
        }
      ]
    },
    output: {
      oneOf: [
        {
          type: 'object',
          contentType: 'application/json',
          properties: {
            response: {
              type: 'string'
            }
          }
        },
        {
          type: 'string',
          contentType: 'text/event-stream',
          format: 'binary'
        }
      ]
    }
  };
  constructor(inputs, modelSettings2) {
    this.inputs = inputs;
    this.modelSettings = modelSettings2 || {
      experimental: true,
      inputsDefaultsStream: {
        max_tokens: 512
      },
      inputsDefaults: {
        max_tokens: 256
      },
      preProcessingArgs: {
        promptTemplate: 'inst',
        defaultContext: ''
      }
    };
  }
  preProcessing() {
    if (this.inputs.stream && this.modelSettings.inputsDefaultsStream) {
      this.inputs = {
        ...this.modelSettings.inputsDefaultsStream,
        ...this.inputs
      };
    } else if (this.modelSettings.inputsDefaults) {
      this.inputs = { ...this.modelSettings.inputsDefaults, ...this.inputs };
    }
    let prompt = '';
    if (this.inputs.messages === void 0) {
      if (this.inputs.raw == true) {
        prompt = this.inputs.prompt;
      } else {
        prompt = generateTgTemplate(
          [
            {
              role: 'system',
              content: this.modelSettings.preProcessingArgs.defaultContext
            },
            { role: 'user', content: this.inputs.prompt }
          ],
          this.modelSettings.preProcessingArgs.promptTemplate
        );
      }
    } else {
      prompt = generateTgTemplate(
        this.inputs.messages,
        this.modelSettings.preProcessingArgs.promptTemplate
      );
    }
    this.preProcessedInputs = {
      prompt,
      max_tokens: this.inputs.max_tokens,
      stream: this.inputs.stream ? true : false
    };
  }
  generateTensors(preProcessedInputs) {
    if (this.modelSettings.generateTensorsFunc) {
      return this.modelSettings.generateTensorsFunc(preProcessedInputs);
    } else {
      return [
        new Tensor('str' /* String */, [preProcessedInputs.prompt], {
          shape: [1],
          name: 'INPUT_0'
        }),
        new Tensor('uint32' /* Uint32 */, [preProcessedInputs.max_tokens], {
          // sequence length
          shape: [1],
          name: 'INPUT_1'
        })
      ];
    }
  }
  postProcessing(response) {
    if (this.modelSettings.postProcessingFunc) {
      this.postProcessedOutputs = {
        response: this.modelSettings.postProcessingFunc(
          response,
          this.preProcessedInputs
        )
      };
    } else {
      this.postProcessedOutputs = { response: response.name.value[0] };
    }
  }
  postProcessingStream(response, inclen) {
    if (this.modelSettings.postProcessingFuncStream) {
      return {
        response: this.modelSettings.postProcessingFuncStream(
          response,
          this.preProcessedInputs,
          inclen
        )
      };
    } else {
      return { response: response.name.value[0] };
    }
  }
};

// ../worker-constellation-entry/src/ai/tasks/text-classification.ts
var AiTextClassification = class {
  modelSettings;
  inputs;
  preProcessedInputs;
  postProcessedOutputs;
  tensors;
  // run ./scripts/gen-validators.ts if you change the schema
  schema = {
    input: {
      type: 'object',
      properties: {
        text: {
          type: 'string'
        }
      },
      required: ['text']
    },
    output: {
      type: 'array',
      contentType: 'application/json',
      items: {
        type: 'object',
        properties: {
          score: {
            type: 'number'
          },
          label: {
            type: 'string'
          }
        }
      }
    }
  };
  constructor(inputs, modelSettings2) {
    this.inputs = inputs;
    this.modelSettings = modelSettings2;
  }
  preProcessing() {
    this.preProcessedInputs = this.inputs;
  }
  generateTensors(preProcessedInputs) {
    if (this.modelSettings.generateTensorsFunc) {
      return this.modelSettings.generateTensorsFunc(preProcessedInputs);
    } else {
      return [
        new Tensor('str' /* String */, [preProcessedInputs.text], {
          shape: [1],
          name: 'input_text'
        })
      ];
    }
  }
  postProcessing(response) {
    this.postProcessedOutputs = [
      {
        label: 'NEGATIVE',
        score: response.logits.value[0][0]
      },
      {
        label: 'POSITIVE',
        score: response.logits.value[0][1]
      }
    ];
  }
};

// ../worker-constellation-entry/src/ai/tasks/text-embeddings.ts
var AiTextEmbeddings = class {
  modelSettings;
  inputs;
  preProcessedInputs;
  postProcessedOutputs;
  tensors;
  // run ./scripts/gen-validators.ts if you change the schema
  schema = {
    input: {
      type: 'object',
      properties: {
        text: {
          oneOf: [
            { type: 'string' },
            {
              type: 'array',
              items: {
                type: 'string'
              },
              maxItems: 100
            }
          ]
        }
      },
      required: ['text']
    },
    output: {
      type: 'object',
      contentType: 'application/json',
      properties: {
        shape: {
          type: 'array',
          items: {
            type: 'number'
          }
        },
        data: {
          type: 'array',
          items: {
            type: 'array',
            items: {
              type: 'number'
            }
          }
        }
      }
    }
  };
  constructor(inputs, modelSettings2) {
    this.inputs = inputs;
    this.modelSettings = modelSettings2;
  }
  preProcessing() {
    this.preProcessedInputs = this.inputs;
  }
  generateTensors(preProcessedInputs) {
    if (this.modelSettings.generateTensorsFunc) {
      return this.modelSettings.generateTensorsFunc(preProcessedInputs);
    } else {
      return [
        new Tensor(
          'str' /* String */,
          Array.isArray(preProcessedInputs.text)
            ? preProcessedInputs.text
            : [preProcessedInputs.text],
          {
            shape: [
              Array.isArray(preProcessedInputs.text)
                ? preProcessedInputs.text.length
                : [preProcessedInputs.text].length
            ],
            name: 'input_text'
          }
        )
      ];
    }
  }
  postProcessing(response) {
    if (this.modelSettings.postProcessingFunc) {
      this.postProcessedOutputs = this.modelSettings.postProcessingFunc(
        response,
        this.preProcessedInputs
      );
    } else {
      this.postProcessedOutputs = {
        shape: response.embeddings.shape,
        data: response.embeddings.value
      };
    }
  }
};
