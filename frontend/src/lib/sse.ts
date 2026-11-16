/** Minimal Server-Sent Events parser for fetch() streams (EventSource can't POST). */

export interface SSEMessage {
  event: string;
  data: string;
}

export interface SSEParser {
  /** Feed decoded text; complete events are dispatched synchronously. */
  push(chunk: string): void;
  /** Flush a final event that was not followed by a blank line. */
  end(): void;
}

export function createSSEParser(onMessage: (message: SSEMessage) => void): SSEParser {
  let buffer = '';
  let event = '';
  let data: string[] = [];

  const dispatch = () => {
    if (data.length > 0) onMessage({ event: event || 'message', data: data.join('\n') });
    event = '';
    data = [];
  };

  const processLine = (line: string) => {
    if (line === '') {
      dispatch();
      return;
    }
    if (line.startsWith(':')) return; // comment / keep-alive
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') event = value;
    else if (field === 'data') data.push(value);
    // 'id' and 'retry' are not used.
  };

  return {
    push(chunk) {
      buffer += chunk;
      for (;;) {
        const match = /\r\n|\r|\n/.exec(buffer);
        if (!match) break;
        // A trailing "\r" may be the first half of "\r\n": wait for the next chunk.
        if (match[0] === '\r' && match.index === buffer.length - 1) break;
        processLine(buffer.slice(0, match.index));
        buffer = buffer.slice(match.index + match[0].length);
      }
    },
    end() {
      if (buffer) processLine(buffer.replace(/\r$/, ''));
      buffer = '';
      dispatch();
    },
  };
}
