#include <Engine.h>

int main(int argc, char **argv)
{
    Engine& engine = Engine::instance();
    engine.init(argc, argv, "tests/engine", 800, 600);
    engine.loadResources("");
    engine.start();
    return 0;
}
