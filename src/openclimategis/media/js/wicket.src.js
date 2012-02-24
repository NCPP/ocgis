/**
 * Inspired by and much borrowed from Chris Pietshmann's mini-library for
 *  converting between Bring Maps shapes (VEShape) to WKT and back.
 *  http://pietschsoft.com/post/2009/04/04/Virtual-Earth-Shapes-%28VEShape%29-to-WKT-%28Well-Known-Text%29-and-Back-using-JavaScript.aspx
 */
var Wkt = (function() {

    var endsWith, startsWith, trim;

    // private
    endsWith = function(str, sub) {
        return str.substring(str.length - sub.length) == sub;
    }

    // private
    startsWith = function(str, sub) {
        return str.substring(0, sub.length) == sub;
    }

    // private
    trim = function(str, sub) {
        sub = sub || ' '; // Defaults to trimming spaces
        // Trim beginning spaces
        while(startsWith(str, sub)) {
            str = str.substring(1);
        }
        // Trim ending spaces
        while(endsWith(str, sub)) {
            str = str.substring(0, str.length - 1);
        }
        return str;
    };

    return {

        trim: trim,

        Wkt: function(str) {
            var digest, // The "body" (digest) of the WKT message
                type; // A truncated name for the WKT geometry type

            if (str == null) {
                throw "Wkt.Wkt: 'str' parameter cannot be null.";
            }

            if (str.length == 0) {
                throw "Wkt.Wkt: 'str' parameter cannot be an empty string.";
            }

            // Get the Shape Type and list of "Longitude Latitude" location points
            switch (trim(str).substring(0, 5).toLowerCase()) {
                case 'point':
                    type = 'point';
                    digest = trim(str).substring(6, trim(str).length - 1);
                    break;
                case 'lines':
                    type = 'lines';
                    digest = trim(str).substring(11, trim(str).length - 1);
                    break;
                case 'polyg':
                    type = 'polyg';
                    digest = trim(str).substring(9, trim(str).length - 2);
                    break;
                case 'multi':
                    type = 'multi';
                    digest = trim(str).substring(15, trim(str).length - 2);
                    break;
                default:
                    throw "Wkt.Wkt.digest: Unknown WKT Geometry Type";
                    break;
            } // eo switch

            return {

                digest: digest,
                type: type,

                 //The WKT geometry as an Array of WKT point strings
                points: trim(trim(digest, '('), ')').split(','),

                 // The base WKT representation without whitespace
                string: trim(str),

                /**
                 * Return the base WKT representation
                 * @return      {String}    The WKT string
                 */
                getRawString: function() { return str || ''; },

            } // eo return
            
        } // eo Wkt

    } // eo return

}()) // Execute immediately
