// ==UserScript==
// @name        rotation
// @namespace   rotation
// @match       https://florr.io/
// @grant       none
// @version     1.0
// @author      lemonqu
// @description fuck the game gives me no supers so im gonna opensource all my codes and quit
// ==/UserScript==
(function () {
    function rewriteFillText() {
        function getCompatibleCanvas() {
            if (typeof (OffscreenCanvasRenderingContext2D) == 'undefined') {
                return [CanvasRenderingContext2D];
            }
            return [OffscreenCanvasRenderingContext2D, CanvasRenderingContext2D];
        }
        const idSymbol = Symbol('id');
        const currentFrameTextList = [];
        for (const { prototype } of getCompatibleCanvas()) {
            prototype[idSymbol] = prototype.fillText;
        }
        for (const { prototype } of getCompatibleCanvas()) {
            prototype.fillText = function (text, x, y) {
                let fontSize = parseFloat(this.font.match(/\d+(\.\d+)?/)[0]);
                if (text == "Mythic" && fontSize == 12) {
                    let { e: x_pos, f: y_pos } = this.getTransform();
                    const canvasWidth = this.canvas ? this.canvas.width : 0;
                    const canvasHeight = this.canvas ? this.canvas.height : 0;
                    currentFrameTextList.push({
                        text: text,
                        x: x_pos,
                        y: y_pos,
                        size: fontSize,
                        color: this.fillStyle,
                        canvasWidth: canvasWidth,
                        canvasHeight: canvasHeight
                    });
                }
                return this[idSymbol](text, x, y);
            };
            prototype.fillText.toString = () => 'function toString() { [native code] }';
        }
        function processFrame() {
            if (currentFrameTextList.length > 0) {
                fetch('http://127.0.0.1:8000/textlist', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ textList: currentFrameTextList })
                }).catch(e => { });
                currentFrameTextList.length = 0;
            }
            requestAnimationFrame(processFrame);
        }
        requestAnimationFrame(processFrame);
    }
    rewriteFillText();
})();
