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
