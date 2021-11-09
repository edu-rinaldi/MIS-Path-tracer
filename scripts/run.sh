./bin/ypathtrace --scene tests/01_cornellbox/cornellbox.json --output out/naive/01_cornellbox_512_256.jpg --shader naive --samples 256 --resolution 512 --bounces 4
./bin/ypathtrace --scene tests/02_matte/matte.json --output out/naive/02_matte_720_256.jpg --shader naive --samples 256 --resolution 720 --bounces 4
./bin/ypathtrace --scene tests/03_metal/metal.json --output out/naive/03_metal_720_256.jpg --shader naive --samples 256 --resolution 720 --bounces 4
./bin/ypathtrace --scene tests/04_plastic/plastic.json --output out/naive/04_plastic_720_256.jpg --shader naive --samples 256 --resolution 720 --bounces 4
./bin/ypathtrace --scene tests/05_glass/glass.json --output out/naive/05_glass_720_256.jpg --shader naive --samples 256 --resolution 720 --bounces 8
./bin/ypathtrace --scene tests/06_opacity/opacity.json --output out/naive/06_opacity_720_256.jpg --shader naive --samples 256 --resolution 720 --bounces 8
./bin/ypathtrace --scene tests/07_hair/hair.json --output out/naive/07_hair_720_256.jpg --shader naive --samples 256 --resolution 720 --bounces 4
./bin/ypathtrace --scene tests/08_lens/lens.json --output out/naive/08_lens_720_256.jpg --shader naive --samples 256 --resolution 720 --bounces 4

./bin/ypathtrace --scene tests/01_cornellbox/cornellbox.json --output out/path/01_cornellbox_512_256.jpg --shader pathtrace --samples 256 --resolution 512 --bounces 4
./bin/ypathtrace --scene tests/02_matte/matte.json --output out/path/02_matte_720_256.jpg --shader pathtrace --samples 256 --resolution 720 --bounces 4
./bin/ypathtrace --scene tests/03_metal/metal.json --output out/path/03_metal_720_256.jpg --shader pathtrace --samples 256 --resolution 720 --bounces 4
./bin/ypathtrace --scene tests/04_plastic/plastic.json --output out/path/04_plastic_720_256.jpg --shader pathtrace --samples 256 --resolution 720 --bounces 4
./bin/ypathtrace --scene tests/05_glass/glass.json --output out/path/05_glass_720_256.jpg --shader pathtrace --samples 256 --resolution 720 --bounces 8
./bin/ypathtrace --scene tests/06_opacity/opacity.json --output out/path/06_opacity_720_256.jpg --shader pathtrace --samples 256 --resolution 720 --bounces 8
./bin/ypathtrace --scene tests/07_hair/hair.json --output out/path/07_hair_720_256.jpg --shader pathtrace --samples 256 --resolution 720 --bounces 4
./bin/ypathtrace --scene tests/08_lens/lens.json --output out/path/08_lens_720_256.jpg --shader pathtrace --samples 256 --resolution 720 --bounces 4

./bin/ypathtrace --scene tests/11_bathroom1/bathroom1.json --output out/path/11_bathroom1_720_256.jpg --shader pathtrace --samples 256 --resolution 720 --bounces 8
./bin/ypathtrace --scene tests/12_ecosys/ecosys.json --output out/path/12_ecosys_720_256.jpg --shader pathtrace --samples 256 --resolution 720 --bounces 8
./bin/ypathtrace --scene tests/13_bedroom/bedroom.json --output out/path/13_bedroom_720_256.jpg --shader pathtrace --samples 256 --resolution 720 --bounces 8
./bin/ypathtrace --scene tests/14_car1/car1.json --output out/path/14_car1_720_256.jpg --shader pathtrace --samples 256 --resolution 720 --bounces 8
./bin/ypathtrace --scene tests/15_classroom/classroom.json --output out/path/15_classroom_720_256.jpg --shader pathtrace --samples 256 --resolution 720 --bounces 8
./bin/ypathtrace --scene tests/16_coffee/coffee.json --output out/path/16_coffee_720_256.jpg --shader pathtrace --samples 256 --resolution 720 --bounces 8
./bin/ypathtrace --scene tests/17_kitchen/kitchen.json --output out/path/17_kitchen_720_256.jpg --shader pathtrace --samples 256 --resolution 720 --bounces 8
