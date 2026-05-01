import { HashMap } from "@aardworx/wombat.adaptive";
let m = HashMap.empty();
m = m.add("outColor", 1);
m = m.add("pickId", 2);
for (const [k, v] of m) console.log(k, v);
