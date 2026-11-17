//! Gemma 4's chat format: messages (and tools) into the prompt text, and the model's answer back into text and tool calls.
//!
//! Tools are declared in the system turn as JSON between `<tools>` tags; the model answers a call as
//! `<tool_call>{"name": …, "arguments": {…}}</tool_call>` (or in Gemma's native `<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>`);
//! the result of a call goes back in a user turn between `<tool_response>` tags.

use std::collections::HashSet;

use serde_json::{Map, Value, json};

/// Closes the model's thought before it starts, so it answers at once. The cached conversation and the next prompt both carry it,
/// which is what lets a follow-up reuse the cache.
pub const EMPTY_THOUGHT: &str = "<|channel>thought\n<channel|>";

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    /// The arguments as a JSON object, serialized (the OpenAI API carries them as a string).
    pub arguments: String,
}

/// The text of a message: a string, or the text parts of a list. Anything else a part may carry (an image) is an error.
fn message_text(message: &Value) -> Result<String, String> {
    match message.get("content") {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => text.push_str(part.get("text").and_then(Value::as_str).unwrap_or("")),
                    Some(other) => {
                        return Err(format!("content of type `{other}` is not supported yet: this server reads text (images need the vision encoder, which is not in the standalone server)"));
                    }
                    None => return Err("a content part has no type".to_owned()),
                }
            }
            Ok(text)
        }
        Some(_) => Err("message content must be a string or a list of parts".to_owned()),
    }
}

/// The prompt for `messages`, with `tools` declared. `thinking` lets the model reason first; otherwise its thought is closed empty.
pub fn render(messages: &[Value], tools: &[Value], thinking: bool) -> Result<String, String> {
    let role = |message: &Value| message.get("role").and_then(Value::as_str).unwrap_or("").to_owned();
    let mut prompt = String::new();
    let first_is_system = messages.first().is_some_and(|message| role(message) == "system");
    if !tools.is_empty() {
        prompt.push_str("<|turn>system\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>");
        for tool in tools {
            prompt.push('\n');
            prompt.push_str(&tool.to_string());
        }
        prompt.push_str("\n</tools>\n\nWhen you need to call a function, you MUST return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\nDo NOT just write the JSON object in plain text or say you called a function without wrapping it in <tool_call> tags. When a tool result answers the question, answer in words and do not call again.\n");
        if first_is_system {
            prompt.push('\n');
            prompt.push_str(&message_text(&messages[0])?);
        }
        prompt.push_str("<turn|>\n");
    }
    let mut index = usize::from(!tools.is_empty() && first_is_system);
    while index < messages.len() {
        let message = &messages[index];
        match role(message).as_str() {
            kind @ ("system" | "user") => {
                prompt.push_str(&format!("<|turn>{kind}\n{}<turn|>\n", message_text(message)?));
                index += 1;
            }
            "assistant" => {
                prompt.push_str("<|turn>model\n");
                if !thinking {
                    prompt.push_str(EMPTY_THOUGHT);
                }
                prompt.push_str(&message_text(message)?);
                for call in message.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
                    let function = call.get("function").ok_or("a tool call has no function")?;
                    let name = function.get("name").and_then(Value::as_str).ok_or("a tool call has no name")?;
                    let arguments = match function.get("arguments") {
                        Some(Value::String(text)) => serde_json::from_str::<Value>(text).map_err(|_| format!("the arguments of `{name}` are not valid JSON"))?,
                        Some(other) => other.clone(),
                        None => json!({}),
                    };
                    prompt.push_str(&format!("\n<tool_call>\n{}\n</tool_call>", json!({ "name": name, "arguments": arguments })));
                }
                prompt.push_str("<turn|>\n");
                index += 1;
            }
            "tool" => {
                prompt.push_str("<|turn>user");
                while index < messages.len() && role(&messages[index]) == "tool" {
                    prompt.push_str(&format!("\n<tool_response>\n{}\n</tool_response>", message_text(&messages[index])?));
                    index += 1;
                }
                prompt.push_str("<turn|>\n");
            }
            other => return Err(format!("unsupported message role `{other}`")),
        }
    }
    prompt.push_str("<|turn>model\n");
    if !thinking {
        prompt.push_str(EMPTY_THOUGHT);
    }
    Ok(prompt)
}

/// The text the model wrote without an empty thought in front of it and without the markers that end a turn.
pub fn clean(text: &str) -> String {
    let text = text.trim_start();
    let text = text.strip_prefix(EMPTY_THOUGHT).unwrap_or(text);
    text.replace("<turn|>", "").trim().to_owned()
}

/// Splits what the model wrote into its words and the tool calls it made (only calls to a declared tool count).
pub fn parse_answer(text: &str, allowed: &HashSet<String>) -> (String, Vec<ToolCall>) {
    let text = clean(text);
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut rest = text.as_str();
    loop {
        let xml = rest.find("<tool_call>");
        let native = rest.find("<|tool_call>");
        let (start, open, close, is_native) = match (xml, native) {
            (Some(x), Some(n)) if n < x => (n, "<|tool_call>", "<tool_call|>", true),
            (Some(x), _) => (x, "<tool_call>", "</tool_call>", false),
            (None, Some(n)) => (n, "<|tool_call>", "<tool_call|>", true),
            (None, None) => break,
        };
        let after = &rest[start + open.len()..];
        let (body, next) = match after.find(close) {
            Some(end) => (&after[..end], &after[end + close.len()..]),
            None => (after, ""),
        };
        let call = if is_native { parse_native_call(body) } else { parse_json_call(body) };
        match call {
            Some(call) if allowed.contains(&call.name) => {
                content.push_str(&rest[..start]);
                calls.push(call);
            }
            // Not a call to a declared tool: it stays in the text.
            _ => content.push_str(&rest[..start + open.len() + body.len() + if after.contains(close) { close.len() } else { 0 }]),
        }
        rest = next;
    }
    content.push_str(rest);
    (content.trim().to_owned(), calls)
}

fn parse_json_call(body: &str) -> Option<ToolCall> {
    let value: Value = serde_json::from_str(body.trim()).ok()?;
    let name = value.get("name")?.as_str()?.to_owned();
    let arguments = match value.get("arguments") {
        Some(Value::String(text)) => serde_json::from_str::<Value>(text).ok()?,
        Some(other) => other.clone(),
        None => json!({}),
    };
    arguments.is_object().then(|| ToolCall { name, arguments: arguments.to_string() })
}

fn parse_native_call(body: &str) -> Option<ToolCall> {
    let call = body.trim().strip_prefix("call:")?;
    let brace = call.find('{')?;
    let arguments = parse_native_value(&call[brace..])?;
    arguments.is_object().then(|| ToolCall { name: call[..brace].trim().to_owned(), arguments: arguments.to_string() })
}

/// Gemma's native values: `{key:<|"|>text<|"|>, n:3, list:[1,2], inner:{…}}` (keys without quotes, strings between `<|"|>` marks).
pub fn parse_native_value(source: &str) -> Option<Value> {
    struct Reader<'a> {
        text: &'a str,
        at: usize,
    }
    const QUOTE: &str = "<|\"|>";
    impl Reader<'_> {
        fn rest(&self) -> &str {
            &self.text[self.at..]
        }
        fn blanks(&mut self) {
            self.at += self.rest().len() - self.rest().trim_start().len();
        }
        fn value(&mut self) -> Option<Value> {
            self.blanks();
            if self.rest().starts_with(QUOTE) {
                self.at += QUOTE.len();
                let end = self.rest().find(QUOTE)?;
                let value = self.rest()[..end].to_owned();
                self.at += end + QUOTE.len();
                return Some(Value::String(value));
            }
            match self.rest().chars().next()? {
                '{' => {
                    self.at += 1;
                    let mut object = Map::new();
                    loop {
                        self.blanks();
                        if self.rest().starts_with('}') {
                            self.at += 1;
                            return Some(Value::Object(object));
                        }
                        let colon = self.rest().find(':')?;
                        let key = self.rest()[..colon].trim().trim_matches('"').to_owned();
                        self.at += colon + 1;
                        object.insert(key, self.value()?);
                        self.blanks();
                        if self.rest().starts_with(',') {
                            self.at += 1;
                        }
                    }
                }
                '[' => {
                    self.at += 1;
                    let mut items = Vec::new();
                    loop {
                        self.blanks();
                        if self.rest().starts_with(']') {
                            self.at += 1;
                            return Some(Value::Array(items));
                        }
                        items.push(self.value()?);
                        self.blanks();
                        if self.rest().starts_with(',') {
                            self.at += 1;
                        }
                    }
                }
                _ => {
                    let end = self.rest().find([',', '}', ']']).unwrap_or(self.rest().len());
                    let word = self.rest()[..end].trim().to_owned();
                    self.at += end;
                    Some(match word.as_str() {
                        "true" => Value::Bool(true),
                        "false" => Value::Bool(false),
                        "null" => Value::Null,
                        other => other.parse::<serde_json::Number>().map(Value::Number).unwrap_or_else(|_| Value::String(other.trim_matches('"').to_owned())),
                    })
                }
            }
        }
    }
    Reader { text: source, at: 0 }.value()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allowed(names: &[&str]) -> HashSet<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    #[test]
    fn a_plain_conversation_ends_with_the_model_turn_and_a_closed_thought() {
        let messages = vec![json!({"role": "system", "content": "Be brief."}), json!({"role": "user", "content": "Hi"})];
        let prompt = render(&messages, &[], false).unwrap();
        assert_eq!(prompt, format!("<|turn>system\nBe brief.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n{EMPTY_THOUGHT}"));
    }

    #[test]
    fn tools_go_in_the_system_turn_and_results_in_a_user_turn() {
        let tools = vec![json!({"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}})];
        let messages = vec![
            json!({"role": "user", "content": "Weather in Paris?"}),
            json!({"role": "assistant", "content": null, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "weather", "arguments": "{\"city\":\"Paris\"}"}}]}),
            json!({"role": "tool", "tool_call_id": "1", "content": "18 C"}),
        ];
        let prompt = render(&messages, &tools, false).unwrap();
        assert!(prompt.starts_with("<|turn>system\n# Tools"));
        assert!(prompt.contains("<tool_call>\n{\"arguments\":{\"city\":\"Paris\"},\"name\":\"weather\"}\n</tool_call>"));
        assert!(prompt.contains("<|turn>user\n<tool_response>\n18 C\n</tool_response><turn|>"));
    }

    #[test]
    fn an_image_is_refused_with_a_reason() {
        let messages = vec![json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]})];
        assert!(render(&messages, &[], false).unwrap_err().contains("vision"));
    }

    #[test]
    fn calls_are_read_in_both_formats_and_undeclared_ones_stay_text() {
        let (content, calls) = parse_answer("Let me check.<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>", &allowed(&["weather"]));
        assert_eq!(content, "Let me check.");
        assert_eq!(calls, vec![ToolCall { name: "weather".into(), arguments: "{\"city\":\"Paris\"}".into() }]);
        let (_, calls) = parse_answer("<|tool_call>call:lookup{city:<|\"|>Paris<|\"|>, days:3, tags:[1,true]}<tool_call|>", &allowed(&["lookup"]));
        assert_eq!(calls[0].name, "lookup");
        assert_eq!(serde_json::from_str::<Value>(&calls[0].arguments).unwrap(), json!({"city": "Paris", "days": 3, "tags": [1, true]}));
        let (content, calls) = parse_answer("<tool_call>{\"name\": \"other\", \"arguments\": {}}</tool_call>", &allowed(&["weather"]));
        assert!(calls.is_empty());
        assert!(content.contains("other"));
    }

    #[test]
    fn the_empty_thought_in_front_of_an_answer_is_dropped() {
        assert_eq!(clean(&format!("{EMPTY_THOUGHT}Paris.<turn|>")), "Paris.");
    }
}
