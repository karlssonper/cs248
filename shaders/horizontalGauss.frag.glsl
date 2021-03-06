uniform sampler2D bloomTex;
uniform sampler2D cocTex;
uniform sampler2D hudTex;

uniform float useHUD;
uniform float texDx;
uniform float focalPlane;

varying vec2 texcoord;

float distance(vec2 tcoords, vec2 uv)
{
    float dx = tcoords.x - uv.x;
    float dy = tcoords.y - uv.y;
    return sqrt(dx*dx + dy*dy);
}

vec3 gaussianBlur(sampler2D tex, int n)
{
    vec3 sum = vec3(0.0);
    float u = texcoord.x - float(n)/2.0 * texDx;
    float totWeight = 0.0;
    for (int i = 0; i < n; ++i, u += texDx){
        float d = distance(texcoord, vec2(u,texcoord.y ));
        float weight = exp(-d*d);
        sum += weight*texture2D(tex, vec2(u,texcoord.y )).rgb;
        totWeight += weight;
    }
    return sum / totWeight;
}

void main() {
	//First color buffer
	gl_FragData[0] = vec4(gaussianBlur(bloomTex,10),1);

	//gl_FragData[1] = vec4(gaussianBlur(cocTex,10),1);

	vec4 hud = texture2D(hudTex, texcoord);

	vec4 coc =texture2D(cocTex, texcoord);;
	if (useHUD > 0.0){
	    if (hud.a > 0) {
            if (focalPlane > 80){
                coc.r = 1.0f;
            } else {
                float t = focalPlane/80.0f;
                coc.r = t*t*t*t;
            }
        }
	}
	gl_FragData[1] = coc;
	//gl_FragData[1] = vec4(1,0,0,1);
}
