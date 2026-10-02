#include "server-common.h"

#include <stdexcept>
#include <string>

int main() {
    server_chat_params opt{};
    opt.use_jinja = true;
    opt.tmpls = common_chat_templates_init(nullptr,
        "{% for message in messages %}{{ message['content'] }}{% endfor %}");

    const json image_request = json::parse(R"({
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jvCAAAAAASUVORK5CYII="}}
        ]}]
    })");
    const std::string unsupported = "image input is not supported - hint: if this is unexpected, you may need to provide the mmproj";

    for (bool retained_history : {false, true}) {
        json body = image_request;
        if (retained_history) {
            body["messages"].push_back({{"role", "assistant"}, {"content", "OK"}});
            body["messages"].push_back({{"role", "user"}, {"content", "Do not use images again. Reply OK."}});
        }
        std::vector<raw_buffer> files;
        bool rejected = false;
        try {
            oaicompat_chat_params_parse(body, opt, files);
        } catch (const std::invalid_argument & e) {
            GGML_ASSERT(e.what() == unsupported);
            rejected = true;
        }
        GGML_ASSERT(rejected);
        GGML_ASSERT(files.empty());
    }

    json text_request = {{"messages", {{{"role", "user"}, {"content", "Reply OK."}}}}};
    std::vector<raw_buffer> files;
    const auto text_params = oaicompat_chat_params_parse(text_request, opt, files);
    GGML_ASSERT(text_params.at("prompt").get<std::string>().find("Reply OK.") != std::string::npos);
    GGML_ASSERT(files.empty());

    opt.allow_image = true;
    json body = image_request;
    const auto vision_params = oaicompat_chat_params_parse(body, opt, files);
    GGML_ASSERT(files.size() == 1);
    GGML_ASSERT(files.front().size() > 8);
    GGML_ASSERT(files.front()[0] == 0x89 && files.front()[1] == 'P');
    GGML_ASSERT(vision_params.at("prompt").get<std::string>().find(get_media_marker()) != std::string::npos);
    GGML_ASSERT(body["messages"][0]["content"][1]["type"] == "media_marker");
    return 0;
}
